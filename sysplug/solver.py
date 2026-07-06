"""Constrained hyperparameter optimisation solver.

The :class:`ConfigSolver` takes an initial training configuration, GPU
hardware constraints, and memory/throughput model predictions, then returns
the best feasible :class:`~sysplug.config.SysPlugConfig`.

Solver algorithm
----------------
1. **Feasibility check**: does the current config fit in GPU memory?
2. **OOM recovery**: if infeasible, reduce batch_size, then increase
   gradient_accumulation (preserving effective batch), then downgrade
   precision until feasible or all options exhausted.
3. **Throughput improvement**: if feasible but utilisation is low, try
   increasing effective batch via gradient_accumulation or upgrading
   precision (fp16 → bf16).
4. **LR scaling**: whenever effective_batch changes, apply the appropriate
   scaling rule (linear for SFT/supervised; sqrt for large batches or RLHF).
5. **Stability warnings**: emit warnings for known dangerous configurations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sysplug.config import SysPlugConfig
from sysplug.hardware import HardwareSnapshot
from sysplug.memory_model import MemoryModel
from sysplug.throughput_model import ThroughputModel
from sysplug.utils.scaling_rules import (
    linear_lr_scale,
    recommended_lr_rule,
    sqrt_lr_scale,
    warmup_steps_for_batch,
)

# Precision downgrade order for OOM recovery (most → least memory-hungry).
# Stops at bf16: int8/int4 are quantized inference/storage formats, not valid
# formats for the *trainable* weights the solver configures. Below bf16 the
# solver relies on gradient checkpointing and smaller batches instead.
_PRECISION_DOWNGRADE: list[str] = ["fp32", "fp16", "bf16"]

# Precision upgrade for throughput (safe upgrades only)
_PRECISION_UPGRADE: dict[str, str] = {"fp32": "bf16", "fp16": "bf16"}


@dataclass
class SolverConstraints:
    """Constraints passed to the solver.

    Attributes:
        memory_safety_factor: Maximum fraction of GPU VRAM that may be used
            (default 0.85 → leave 15% headroom).
        max_grad_accumulation: Upper limit on gradient accumulation steps.
        min_batch_size: Minimum allowable per-device batch size.
    """

    memory_safety_factor: float = 0.85
    max_grad_accumulation: int = 64
    min_batch_size: int = 1


def _effective_batch(batch_size: int, grad_acc: int, gpu_count: int) -> int:
    return batch_size * grad_acc * gpu_count


def _is_feasible(
    predicted_mb: float,
    available_mb: float,
    safety_factor: float,
) -> bool:
    """Return True if predicted memory fits within the safety budget."""
    budget = available_mb * safety_factor
    return predicted_mb <= budget


class ConfigSolver:
    """Constrained hyperparameter solver.

    Args:
        memory_model: :class:`~sysplug.memory_model.MemoryModel` instance.
        throughput_model: :class:`~sysplug.throughput_model.ThroughputModel` instance.
        constraints: Solver constraints (memory budget, batch limits, etc.).
        training_type: Training regime for LR scaling rule selection.
        objective: Optimisation objective:
            ``"throughput"`` | ``"memory"`` | ``"balanced"``.
        verbose: Log solver decisions to the console.

    Examples:
        >>> from sysplug.hardware import HardwareSnapshot, GPUSnapshot
        >>> from sysplug.memory_model import MemoryModel
        >>> from sysplug.throughput_model import ThroughputModel
        >>> hw = HardwareSnapshot(gpus=[
        ...     GPUSnapshot(0, "A100", 40960, 0, 40960, 0, 0, (8,0), 2039)
        ... ])
        >>> solver = ConfigSolver(MemoryModel(), ThroughputModel())
        >>> config = {"batch_size": 4, "learning_rate": 1e-4,
        ...           "precision": "bf16", "optimizer": "adamw",
        ...           "parallelism": "none",
        ...           "use_gradient_checkpointing": False,
        ...           "gradient_accumulation": 1}
        >>> result = solver.solve(config, hw, param_count=125_000_000)
        >>> isinstance(result, SysPlugConfig)
        True
    """

    def __init__(
        self,
        memory_model: MemoryModel,
        throughput_model: ThroughputModel,
        constraints: Optional[SolverConstraints] = None,
        training_type: str = "supervised",
        objective: str = "balanced",
        verbose: bool = True,
    ) -> None:
        self._mem_model = memory_model
        self._tput_model = throughput_model
        self._constraints = constraints or SolverConstraints()
        self._training_type = training_type
        self._objective = objective
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        config: Dict[str, Any],
        hardware: HardwareSnapshot,
        param_count: int,
        sequence_length: int = 512,
        locked_params: Optional[Dict[str, Any]] = None,
    ) -> SysPlugConfig:
        """Produce a feasible, optimised :class:`SysPlugConfig`.

        Args:
            config: Initial configuration dict.  Must contain at minimum:
                ``batch_size``, ``learning_rate``, ``precision``,
                ``optimizer``, ``parallelism``,
                ``use_gradient_checkpointing``, ``gradient_accumulation``.
            hardware: Current hardware snapshot from
                :class:`~sysplug.hardware.HardwareProfiler`.
            param_count: Model parameter count.
            sequence_length: Token sequence length for memory/throughput estimates.
            locked_params: Parameters that must not be changed by the solver
                (used by the ``what_if`` engine).

        Returns:
            A :class:`SysPlugConfig` with recommended values.

        Raises:
            ValueError: If the config dict is missing required keys.
        """
        cfg = self._normalise_config(config)
        locked = set(locked_params or {})
        warnings: List[str] = []
        notes: List[str] = []

        # Available memory per GPU
        if hardware.gpus:
            available_mb = hardware.gpus[0].total_memory_mb
            gpu_count = hardware.gpu_count
        else:
            # CPU-only fallback: use generous budget
            available_mb = 200_000.0
            gpu_count = 1

        original_eff_batch = _effective_batch(
            cfg["batch_size"], cfg["gradient_accumulation"], gpu_count
        )
        original_lr = cfg["learning_rate"]

        # ------------------------------------------------------------------
        # Step 1: Feasibility check → OOM recovery
        # ------------------------------------------------------------------
        cfg, oom_notes = self._ensure_feasible(
            cfg, available_mb, param_count, gpu_count, sequence_length, locked
        )
        notes.extend(oom_notes)

        # ------------------------------------------------------------------
        # Step 2: Throughput improvement (if objective is not "memory")
        # ------------------------------------------------------------------
        if self._objective in {"throughput", "balanced"}:
            cfg, tput_notes = self._improve_throughput(
                cfg, available_mb, param_count, gpu_count, sequence_length, locked
            )
            notes.extend(tput_notes)

        # ------------------------------------------------------------------
        # Step 3: LR scaling when effective batch changed
        # ------------------------------------------------------------------
        new_eff_batch = _effective_batch(
            cfg["batch_size"], cfg["gradient_accumulation"], gpu_count
        )
        if new_eff_batch != original_eff_batch and "learning_rate" not in locked:
            cfg["learning_rate"], lr_note = self._scale_lr(
                original_lr, original_eff_batch, new_eff_batch
            )
            notes.append(lr_note)

        # ------------------------------------------------------------------
        # Step 4: Warmup scaling
        # ------------------------------------------------------------------
        if "warmup_steps" in cfg and "warmup_steps" not in locked:
            cfg["warmup_steps"] = warmup_steps_for_batch(
                cfg["warmup_steps"], original_eff_batch, new_eff_batch
            )

        # ------------------------------------------------------------------
        # Step 5: Stability warnings
        # ------------------------------------------------------------------
        warnings.extend(self._stability_warnings(cfg))

        # ------------------------------------------------------------------
        # Final prediction
        # ------------------------------------------------------------------
        mem_est = self._mem_model.predict(
            param_count=param_count,
            batch_size=cfg["batch_size"],
            precision=cfg["precision"],
            optimizer=cfg["optimizer"],
            parallelism=cfg["parallelism"],
            use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
            sequence_length=sequence_length,
        )

        tput_est = self._tput_model.predict(
            effective_batch_size=new_eff_batch,
            model_size_params=param_count,
            precision=cfg["precision"],
            sequence_length=sequence_length,
        )

        return SysPlugConfig(
            batch_size=cfg["batch_size"],
            gradient_accumulation=cfg["gradient_accumulation"],
            effective_batch_size=new_eff_batch,
            learning_rate=cfg["learning_rate"],
            precision=cfg["precision"],
            optimizer=cfg["optimizer"],
            parallelism=cfg["parallelism"],
            use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
            predicted_peak_memory_mb=mem_est.peak_memory_mb,
            predicted_throughput_samples_per_sec=tput_est.samples_per_sec,
            safety_margin_pct=self._constraints.memory_safety_factor,
            warnings=warnings,
            notes=notes,
            solver_objective=self._objective,
            training_type=self._training_type,
            gpu_count=gpu_count,
            sequence_length=sequence_length,
            param_count=param_count,
        )

    # ------------------------------------------------------------------
    # OOM recovery
    # ------------------------------------------------------------------

    def _ensure_feasible(
        self,
        cfg: Dict[str, Any],
        available_mb: float,
        param_count: int,
        gpu_count: int,
        sequence_length: int,
        locked: set[str],
    ) -> tuple[Dict[str, Any], List[str]]:
        """Reduce batch/precision until memory fits within budget."""
        notes: List[str] = []
        c = copy.deepcopy(cfg)

        def predict_mem() -> float:
            est = self._mem_model.predict(
                param_count=param_count,
                batch_size=c["batch_size"],
                precision=c["precision"],
                optimizer=c["optimizer"],
                parallelism=c["parallelism"],
                use_gradient_checkpointing=c["use_gradient_checkpointing"],
                sequence_length=sequence_length,
            )
            return est.peak_memory_mb

        # Try gradient checkpointing first (preserves throughput better)
        if (
            not c["use_gradient_checkpointing"]
            and "use_gradient_checkpointing" not in locked
            and not _is_feasible(
                predict_mem(), available_mb, self._constraints.memory_safety_factor
            )
        ):
            c["use_gradient_checkpointing"] = True
            notes.append("Enabled gradient checkpointing to reduce activation memory.")

        max_iters = 10
        iteration = 0
        while (
            not _is_feasible(predict_mem(), available_mb, self._constraints.memory_safety_factor)
            and iteration < max_iters
        ):
            iteration += 1
            reduced = False

            # Try halving batch_size
            if (
                "batch_size" not in locked
                and c["batch_size"] // 2 >= self._constraints.min_batch_size
            ):
                old_bs = c["batch_size"]
                c["batch_size"] = c["batch_size"] // 2
                # Double grad_acc to preserve effective batch (if allowed)
                if (
                    "gradient_accumulation" not in locked
                    and c["gradient_accumulation"] * 2 <= self._constraints.max_grad_accumulation
                ):
                    c["gradient_accumulation"] *= 2
                notes.append(
                    f"Reduced batch_size {old_bs} → {c['batch_size']} "
                    f"(grad_acc now {c['gradient_accumulation']}) to fit in GPU memory."
                )
                reduced = True

            # Try precision downgrade
            elif "precision" not in locked:
                idx = _PRECISION_DOWNGRADE.index(c["precision"])
                if idx < len(_PRECISION_DOWNGRADE) - 1:
                    old_prec = c["precision"]
                    c["precision"] = _PRECISION_DOWNGRADE[idx + 1]
                    notes.append(
                        f"Downgraded precision {old_prec} → {c['precision']} to fit in GPU memory."
                    )
                    reduced = True

            if not reduced:
                notes.append(
                    "WARNING: Could not find a feasible configuration within constraints. "
                    "Consider using a larger GPU or ZeRO parallelism."
                )
                break

        return c, notes

    # ------------------------------------------------------------------
    # Throughput improvement
    # ------------------------------------------------------------------

    def _improve_throughput(
        self,
        cfg: Dict[str, Any],
        available_mb: float,
        param_count: int,
        gpu_count: int,
        sequence_length: int,
        locked: set[str],
    ) -> tuple[Dict[str, Any], List[str]]:
        """Try to increase throughput while staying within memory budget."""
        notes: List[str] = []
        c = copy.deepcopy(cfg)

        def predict_mem(batch: int, prec: str, grad_acc: int) -> float:
            est = self._mem_model.predict(
                param_count=param_count,
                batch_size=batch,
                precision=prec,
                optimizer=c["optimizer"],
                parallelism=c["parallelism"],
                use_gradient_checkpointing=c["use_gradient_checkpointing"],
                sequence_length=sequence_length,
            )
            return est.peak_memory_mb

        # Upgrade precision if possible (fp16 → bf16 is numerically safer)
        if "precision" not in locked and c["precision"] in _PRECISION_UPGRADE:
            new_prec = _PRECISION_UPGRADE[c["precision"]]
            if _is_feasible(
                predict_mem(c["batch_size"], new_prec, c["gradient_accumulation"]),
                available_mb,
                self._constraints.memory_safety_factor,
            ):
                old_prec = c["precision"]
                c["precision"] = new_prec
                notes.append(f"Upgraded precision {old_prec} → {new_prec} for better throughput.")

        # Increase effective batch via gradient_accumulation
        if (
            "gradient_accumulation" not in locked
            and c["gradient_accumulation"] * 2 <= self._constraints.max_grad_accumulation
        ):
            doubled_ga = c["gradient_accumulation"] * 2
            if _is_feasible(
                predict_mem(c["batch_size"], c["precision"], doubled_ga),
                available_mb,
                self._constraints.memory_safety_factor,
            ):
                c["gradient_accumulation"] = doubled_ga
                notes.append(
                    f"Increased gradient_accumulation to {doubled_ga} "
                    "to improve effective batch size."
                )

        return c, notes

    # ------------------------------------------------------------------
    # LR scaling
    # ------------------------------------------------------------------

    def _scale_lr(
        self,
        original_lr: float,
        original_eff_batch: int,
        new_eff_batch: int,
    ) -> tuple[float, str]:
        """Apply the appropriate LR scaling rule."""
        rule = recommended_lr_rule(self._training_type, new_eff_batch)
        if rule == "linear":
            new_lr = linear_lr_scale(original_lr, original_eff_batch, new_eff_batch)
            note = (
                f"Applied linear LR scaling: {original_lr:.2e} → {new_lr:.2e} "
                f"(batch {original_eff_batch} → {new_eff_batch})."
            )
        else:
            new_lr = sqrt_lr_scale(original_lr, original_eff_batch, new_eff_batch)
            note = (
                f"Applied sqrt LR scaling: {original_lr:.2e} → {new_lr:.2e} "
                f"(batch {original_eff_batch} → {new_eff_batch})."
            )
        return new_lr, note

    # ------------------------------------------------------------------
    # Stability warnings
    # ------------------------------------------------------------------

    @staticmethod
    def _stability_warnings(cfg: Dict[str, Any]) -> List[str]:
        """Emit warnings for known dangerous hyperparameter combinations."""
        warns: List[str] = []

        lr = cfg.get("learning_rate", 0.0)
        precision = cfg.get("precision", "bf16")
        training_type = cfg.get("training_type", "supervised")
        batch_size = cfg.get("batch_size", 8)
        grad_acc = cfg.get("gradient_accumulation", 1)
        grad_clip = cfg.get("max_grad_norm", None)

        # High LR + fp16 without gradient clipping is risky
        if (
            lr > 1e-3
            and precision == "fp16"
            and (grad_clip is None or float(grad_clip) > 5.0)
        ):
            warns.append(
                f"LR {lr:.1e} with fp16 and no gradient clipping is risky. "
                "Set max_grad_norm ≤ 1.0 or switch to bf16."
            )

        # Small batch with RLHF causes high variance in reward estimates
        if training_type == "rlhf" and batch_size < 4:
            warns.append(
                f"batch_size={batch_size} is very small for RLHF; "
                "reward variance will be high. Use batch_size ≥ 4."
            )

        # Very high gradient accumulation introduces staleness
        if grad_acc > 32:
            warns.append(
                f"gradient_accumulation={grad_acc} is high (>32). "
                "This can cause training instability due to gradient staleness."
            )

        return warns

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing config keys with sensible defaults."""
        defaults: Dict[str, Any] = {
            "batch_size": 8,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "bf16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
            "training_type": "supervised",
        }
        merged = {**defaults, **config}
        # Normalise precision to lowercase
        merged["precision"] = str(merged["precision"]).lower()
        return merged
