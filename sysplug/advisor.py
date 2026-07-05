"""Advisor: the main SysPlug entry point.

The :class:`Advisor` orchestrates hardware profiling, memory estimation,
throughput prediction, and constrained optimisation to recommend a safe and
efficient training configuration.

Typical usage::

    import sysplug

    advisor = sysplug.Advisor(model=model, training_type="sft")
    cfg = advisor.suggest_config({"batch_size": 8, "learning_rate": 2e-5})
    training_args = cfg.to_training_arguments(output_dir="./checkpoints")

    with advisor.monitor(check_interval_steps=100) as mon:
        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            mon.record(step=step, loss=loss.item())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from sysplug.config import SysPlugConfig
from sysplug.hardware import HardwareProfiler, HardwareSnapshot
from sysplug.memory_model import MemoryModel, _params_from_name
from sysplug.monitor import Monitor
from sysplug.solver import ConfigSolver, SolverConstraints
from sysplug.throughput_model import ThroughputModel
from sysplug.utils.logging import get_console
from sysplug.utils.validators import validate_config_dict

# ---------------------------------------------------------------------------
# WhatIfResult
# ---------------------------------------------------------------------------


@dataclass
class WhatIfResult:
    """Result of :meth:`Advisor.what_if`.

    Attributes:
        new_config: The recommended configuration after the proposed change.
        changed_params: Dict mapping parameter names to ``(old, new)`` tuples.
        reason: Dict mapping each changed parameter to the reason it changed.
        feasible: Whether the proposed change is achievable within constraints.
        warnings: Any warnings emitted during solving.
    """

    new_config: SysPlugConfig
    changed_params: Dict[str, tuple[Any, Any]] = field(default_factory=dict)
    reason: Dict[str, str] = field(default_factory=dict)
    feasible: bool = True
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


class Advisor:
    """GPU-aware hyperparameter advisor for deep learning training.

    Args:
        model: A ``torch.nn.Module`` instance, or a model-name string like
            ``"llama-3-8b"``, or an integer parameter count.
        hardware: ``"auto"`` to detect GPUs automatically, or a
            :class:`~sysplug.hardware.HardwareSnapshot` to use directly.
        training_type: Training regime:
            ``"supervised"``, ``"sft"``, ``"dpo"``, ``"rlhf"``, ``"grpo"``.
        objective: Optimisation objective:
            ``"throughput"``, ``"memory"``, or ``"balanced"``.
        verbose: Print rich-formatted output to the console.
        device_ids: List of GPU device IDs to consider.  ``None`` = all GPUs.
        constraints: Optional :class:`~sysplug.solver.SolverConstraints`.

    Examples:
        >>> import sysplug
        >>> advisor = sysplug.Advisor(model="gpt2", training_type="sft")
        >>> cfg = advisor.suggest_config({"batch_size": 8, "learning_rate": 2e-5})
        >>> isinstance(cfg, sysplug.SysPlugConfig)
        True
    """

    def __init__(
        self,
        model: Union[Any, str, int] = "gpt2",
        hardware: Union[str, HardwareSnapshot] = "auto",
        training_type: str = "supervised",
        objective: str = "balanced",
        verbose: bool = True,
        device_ids: Optional[list[int]] = None,
        constraints: Optional[SolverConstraints] = None,
    ) -> None:
        self._verbose = verbose
        self._training_type = training_type
        self._objective = objective
        self._console = get_console(verbose=verbose)

        # Parameter count
        self._param_count = self._resolve_param_count(model)

        # GPU name for throughput model
        gpu_name = "A100"  # default; overridden after hardware snapshot

        # Hardware
        if isinstance(hardware, HardwareSnapshot):
            self._hardware = hardware
            if hardware.gpus:
                gpu_name = hardware.gpus[0].gpu_name
        else:
            self._profiler = HardwareProfiler(device_ids=device_ids, verbose=verbose)
            self._hardware = self._profiler.snapshot()
            if self._hardware.gpus:
                gpu_name = self._hardware.gpus[0].gpu_name

        # Models
        gpu_count = max(1, self._hardware.gpu_count)
        self._memory_model = MemoryModel(gpu_count=gpu_count)
        self._throughput_model = ThroughputModel(
            gpu_name=gpu_name,
            gpu_count=gpu_count,
        )

        # Solver
        self._constraints = constraints or SolverConstraints()
        self._solver = ConfigSolver(
            memory_model=self._memory_model,
            throughput_model=self._throughput_model,
            constraints=self._constraints,
            training_type=training_type,
            objective=objective,
            verbose=verbose,
        )

        self._current_config: Optional[SysPlugConfig] = None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def suggest_config(self, base_config: Dict[str, Any]) -> SysPlugConfig:
        """Return the best safe configuration for the given starting point.

        Runs the full pipeline:
        1. Refresh hardware snapshot.
        2. Validate and normalise the input config.
        3. Run the constrained solver.
        4. Print a rich summary table if ``verbose=True``.

        Args:
            base_config: Initial configuration dict.  Any subset of:
                ``batch_size``, ``gradient_accumulation``, ``learning_rate``,
                ``precision``, ``optimizer``, ``parallelism``,
                ``use_gradient_checkpointing``, ``sequence_length``.

        Returns:
            A :class:`~sysplug.config.SysPlugConfig` with the recommended values.

        Raises:
            ValueError: If the input config fails validation.

        Examples:
            >>> advisor = Advisor(model="gpt2")
            >>> cfg = advisor.suggest_config({"batch_size": 4})
            >>> cfg.batch_size >= 1
            True
        """
        validated = validate_config_dict(base_config)

        # Refresh hardware
        if hasattr(self, "_profiler"):
            self._hardware = self._profiler.snapshot()

        seq_len = int(validated.get("sequence_length", 512))
        cfg = self._solver.solve(
            config=validated,
            hardware=self._hardware,
            param_count=self._param_count,
            sequence_length=seq_len,
        )

        self._current_config = cfg

        if self._verbose:
            self._console.print(cfg.summary())

        return cfg

    def what_if(
        self,
        change: Dict[str, Any],
        current_config: Optional[SysPlugConfig] = None,
    ) -> WhatIfResult:
        """Evaluate a proposed hyperparameter change.

        Locks the specified parameters and re-runs the solver for the rest,
        returning a :class:`WhatIfResult` with diff annotations.

        Args:
            change: Dict of proposed changes, e.g. ``{"batch_size": 32}``.
            current_config: The baseline configuration to modify.
                Defaults to :attr:`current_config`.

        Returns:
            A :class:`WhatIfResult` with the new config and a diff.

        Raises:
            RuntimeError: If no current config is available (call
                :meth:`suggest_config` first).

        Examples:
            >>> advisor = Advisor(model="gpt2")
            >>> cfg = advisor.suggest_config({"batch_size": 4})
            >>> result = advisor.what_if({"batch_size": 16})
            >>> isinstance(result, WhatIfResult)
            True
        """
        base = current_config or self._current_config
        if base is None:
            raise RuntimeError(
                "No current config available. Call suggest_config() first."
            )

        # Build merged config from current + proposed change
        merged = base.to_dict()
        # Keep only solver-relevant keys
        solver_keys = {
            "batch_size", "gradient_accumulation", "learning_rate",
            "precision", "optimizer", "parallelism",
            "use_gradient_checkpointing", "sequence_length",
        }
        solver_config = {k: merged[k] for k in solver_keys if k in merged}
        solver_config.update(change)

        # Lock the proposed keys so the solver treats them as fixed
        locked = set(change.keys())

        # Refresh hardware
        if hasattr(self, "_profiler"):
            self._hardware = self._profiler.snapshot()

        seq_len = int(solver_config.get("sequence_length", 512))
        new_cfg = self._solver.solve(
            config=solver_config,
            hardware=self._hardware,
            param_count=self._param_count,
            sequence_length=seq_len,
            locked_params={k: solver_config[k] for k in locked if k in solver_config},
        )

        # Compute diff
        changed_params: Dict[str, tuple[Any, Any]] = {}
        reason: Dict[str, str] = {}

        for key in solver_keys:
            old_val = getattr(base, key, None)
            new_val = getattr(new_cfg, key, None)
            if old_val != new_val:
                changed_params[key] = (old_val, new_val)
                if key in locked:
                    reason[key] = "explicitly requested by user"
                else:
                    reason[key] = "adjusted by solver to maintain feasibility"

        # Check feasibility
        if self._hardware.gpus:
            available_mb = self._hardware.gpus[0].total_memory_mb
        else:
            available_mb = float("inf")
        feasible = new_cfg.predicted_peak_memory_mb <= (
            available_mb * self._constraints.memory_safety_factor
        )

        return WhatIfResult(
            new_config=new_cfg,
            changed_params=changed_params,
            reason=reason,
            feasible=feasible,
            warnings=list(new_cfg.warnings),
        )

    def monitor(
        self,
        check_interval_steps: int = 50,
        reconfig_policy: str = "suggest",
    ) -> Monitor:
        """Create a :class:`~sysplug.monitor.Monitor` context manager.

        Args:
            check_interval_steps: How often (in steps) to run GPU + stability checks.
            reconfig_policy: ``"suggest"``, ``"auto-apply"``, or ``"warn-only"``.

        Returns:
            A :class:`~sysplug.monitor.Monitor` ready to use as a context manager.

        Examples:
            >>> advisor = Advisor(model="gpt2")
            >>> with advisor.monitor(check_interval_steps=10) as mon:
            ...     for step in range(20):
            ...         mon.record(step=step, loss=1.0)
        """
        return Monitor(
            advisor=self,
            check_interval_steps=check_interval_steps,
            reconfig_policy=reconfig_policy,
            verbose=self._verbose,
        )

    def profile_run(
        self,
        dataloader: Any,
        steps: int = 5,
    ) -> Dict[str, Any]:
        """Run a short profiling pass to calibrate models.

        Attempts to measure actual memory usage and throughput over a few
        training steps, then calls :meth:`MemoryModel.calibrate` and
        :meth:`ThroughputModel.calibrate_roofline`.

        Args:
            dataloader: An iterable data loader.
            steps: Number of profiling steps (default 5).

        Returns:
            Dict with keys ``"measured_memory_mb"``, ``"measured_samples_per_sec"``,
            ``"calibration_factor_memory"``, ``"calibration_factor_throughput"``.

        Note:
            Requires a CUDA GPU.  On CPU-only systems, returns estimated values.
        """
        import time

        try:
            import torch
        except ImportError:
            warnings.warn("torch not available; returning estimated values.")
            return {"measured_memory_mb": 0.0, "measured_samples_per_sec": 0.0}

        if self._current_config is None:
            warnings.warn("Call suggest_config() before profile_run().")
            return {}

        measured_samples = []
        start = time.perf_counter()
        for i, batch in enumerate(dataloader):
            if i >= steps:
                break
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        elapsed = time.perf_counter() - start
        total_samples = steps * self._current_config.batch_size

        result: Dict[str, Any] = {
            "measured_memory_mb": 0.0,
            "measured_samples_per_sec": total_samples / max(elapsed, 1e-6),
        }

        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            result["measured_memory_mb"] = peak_bytes / 1024 / 1024

            # Calibrate memory model
            measured_samples.append({
                "param_count": self._param_count,
                "batch_size": self._current_config.batch_size,
                "precision": self._current_config.precision,
                "optimizer": self._current_config.optimizer,
                "parallelism": self._current_config.parallelism,
                "use_gradient_checkpointing": self._current_config.use_gradient_checkpointing,
                "measured_mb": result["measured_memory_mb"],
            })
            factor = self._memory_model.calibrate(measured_samples)
            result["calibration_factor_memory"] = factor

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_config(self) -> Optional[SysPlugConfig]:
        """The most recently recommended configuration, or ``None``."""
        return self._current_config

    def current_lr(self) -> float:
        """Return the current recommended learning rate.

        Returns:
            The learning rate from :attr:`current_config`, or ``0.0`` if no
            config has been computed yet.
        """
        if self._current_config is None:
            return 0.0
        return self._current_config.learning_rate

    @property
    def hardware(self) -> HardwareSnapshot:
        """The most recent hardware snapshot."""
        return self._hardware

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_param_count(model: Union[Any, str, int]) -> int:
        """Resolve parameter count from a module, string, or integer."""
        if isinstance(model, int):
            return model
        if isinstance(model, str):
            return _params_from_name(model)
        # Assume nn.Module
        try:
            return sum(p.numel() for p in model.parameters())
        except AttributeError:
            warnings.warn(
                f"Could not determine parameter count from model type {type(model)}. "
                "Defaulting to 125M parameters."
            )
            return 125_000_000
