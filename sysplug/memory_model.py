"""Analytic GPU memory model for deep learning training.

Implements a static analytical model that estimates peak GPU memory usage
given model architecture, batch size, precision, optimizer, and parallelism
strategy.  The model can be calibrated against real measurements via a
least-squares correction factor.

Memory components:
    - Parameters (P): depends on precision (bytes per element).
    - Gradients (G): same dtype as parameters; sharded under ZeRO-2/3.
    - Optimizer states (O): AdamW = 2×FP32; SGD = 0; Adafactor ≈ 0.5×.
    - Activations (A): proportional to batch_size × seq_len × hidden_dim.
    - Framework overhead (F): fixed ~500 MB.

References:
    - Rajbhandari et al. (2020) "ZeRO: Memory Optimizations Toward Training
      Trillion Parameter Models". https://arxiv.org/abs/1910.02054
    - Hou et al. (2021) "Memory Efficient Adaptive Optimization"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PrecisionMode(str, Enum):
    """Supported training precisions.

    Each member stores the number of bytes required per parameter element.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


_BYTES_PER_PARAM: dict[PrecisionMode, float] = {
    PrecisionMode.FP32: 4.0,
    PrecisionMode.FP16: 2.0,
    PrecisionMode.BF16: 2.0,
    PrecisionMode.INT8: 1.0,
    PrecisionMode.INT4: 0.5,
}

# ---------------------------------------------------------------------------
# Model-name lookup table (approximate param counts in billions)
# ---------------------------------------------------------------------------

_MODEL_PARAM_BILLIONS: dict[str, float] = {
    "gpt2": 0.117,
    "gpt2-medium": 0.345,
    "gpt2-large": 0.774,
    "gpt2-xl": 1.558,
    "llama-2-7b": 7.0,
    "llama-2-13b": 13.0,
    "llama-2-70b": 70.0,
    "llama-3-8b": 8.0,
    "llama-3-70b": 70.0,
    "mistral-7b": 7.0,
    "mixtral-8x7b": 46.7,
    "falcon-7b": 7.0,
    "falcon-40b": 40.0,
    "opt-1.3b": 1.3,
    "opt-6.7b": 6.7,
    "opt-30b": 30.0,
    "bloom-7b1": 7.1,
    "t5-small": 0.060,
    "t5-base": 0.220,
    "t5-large": 0.738,
    "t5-xl": 2.85,
    "t5-xxl": 11.3,
    "bert-base": 0.110,
    "bert-large": 0.340,
    "roberta-base": 0.125,
    "roberta-large": 0.355,
    "phi-2": 2.7,
    "gemma-2b": 2.0,
    "gemma-7b": 7.0,
    "qwen-7b": 7.0,
    "qwen-14b": 14.0,
    "qwen-72b": 72.0,
    "codellama-7b": 7.0,
    "codellama-13b": 13.0,
    "codellama-34b": 34.0,
    "deepseek-7b": 7.0,
    "deepseek-67b": 67.0,
    "starcoder": 15.5,
    "starcoder2-7b": 7.0,
    "starcoder2-15b": 15.0,
}


def _params_from_name(model_name: str) -> int:
    """Approximate parameter count from a model name string.

    Args:
        model_name: Model name like ``"llama-3-8b"`` or ``"gpt2-xl"``.

    Returns:
        Approximate parameter count as an integer.

    Raises:
        ValueError: If the model name is not in the lookup table.
    """
    key = model_name.lower().strip()
    for name_key, billions in _MODEL_PARAM_BILLIONS.items():
        if name_key in key or key in name_key:
            return int(billions * 1e9)
    raise ValueError(
        f"Unknown model name '{model_name}'. Provide param_count directly or "
        f"use one of: {sorted(_MODEL_PARAM_BILLIONS.keys())}"
    )


def _infer_hidden_layers(param_count: int) -> Tuple[int, int]:
    """Infer transformer ``(hidden_size, num_layers)`` from a parameter count.

    Uses the standard decoder relation ``params ≈ 12 · layers · hidden²`` with an
    empirically-anchored aspect ratio ``hidden ≈ 2.1 · params**(1/3)`` (rounded
    to a multiple of 128). Reproduces real configs far better than a flat
    heuristic — e.g. 7B → ~3968 hidden / ~37 layers (real Llama-7B is 4096 / 32),
    70B → ~8704 / ~77 (real 8192 / 80).

    Args:
        param_count: Total parameter count.

    Returns:
        ``(hidden_size, num_layers)``, each at least its sensible floor.
    """
    hidden = max(128, int(round(2.1 * (param_count ** (1.0 / 3.0)) / 128.0)) * 128)
    layers = max(1, int(round(param_count / (12.0 * hidden * hidden))))
    return hidden, layers


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MemoryBreakdown:
    """Breakdown of memory usage by component (all values in MiB).

    Attributes:
        parameters_mb: Memory for model parameters.
        gradients_mb: Memory for parameter gradients.
        optimizer_states_mb: Memory for optimizer state tensors.
        activations_mb: Memory for forward-pass activations.
        framework_overhead_mb: Fixed CUDA/framework overhead.
        total_mb: Sum of all components.
    """

    parameters_mb: float = 0.0
    gradients_mb: float = 0.0
    optimizer_states_mb: float = 0.0
    activations_mb: float = 0.0
    framework_overhead_mb: float = 500.0

    @property
    def total_mb(self) -> float:
        """Total predicted memory in MiB."""
        return (
            self.parameters_mb
            + self.gradients_mb
            + self.optimizer_states_mb
            + self.activations_mb
            + self.framework_overhead_mb
        )

    def to_dict(self) -> Dict[str, float]:
        """Return the breakdown as a plain dictionary."""
        return {
            "parameters_mb": self.parameters_mb,
            "gradients_mb": self.gradients_mb,
            "optimizer_states_mb": self.optimizer_states_mb,
            "activations_mb": self.activations_mb,
            "framework_overhead_mb": self.framework_overhead_mb,
            "total_mb": self.total_mb,
        }


@dataclass
class MemoryEstimate:
    """Result of :meth:`MemoryModel.predict`.

    Attributes:
        peak_memory_mb: Central estimate of peak memory in MiB.
        lower_mb: Lower bound of the 90% confidence interval.
        upper_mb: Upper bound of the 90% confidence interval.
        breakdown: Per-component memory breakdown.
        gpu_count: Number of GPUs assumed in the estimate.
        calibration_factor: Correction factor applied (1.0 = uncalibrated).
    """

    peak_memory_mb: float
    lower_mb: float
    upper_mb: float
    breakdown: MemoryBreakdown
    gpu_count: int = 1
    calibration_factor: float = 1.0


# ---------------------------------------------------------------------------
# MemoryModel
# ---------------------------------------------------------------------------


class MemoryModel:
    """Analytic peak-GPU-memory predictor for DL training.

    Implements a static formula-based model covering all major training
    configurations.  Can optionally be calibrated against real measurements.

    Args:
        gpu_count: Number of GPUs used for distributed training.
        calibration_factor: Multiplicative correction applied to the raw
            analytic estimate.  ``1.0`` means no correction.

    Examples:
        >>> model = MemoryModel(gpu_count=1)
        >>> est = model.predict(
        ...     param_count=7_000_000_000,
        ...     batch_size=4,
        ...     precision="bf16",
        ...     optimizer="adamw",
        ...     parallelism="none",
        ...     use_gradient_checkpointing=False,
        ... )
        >>> est.peak_memory_mb > 0
        True
    """

    # Confidence interval half-widths (empirically derived)
    _CI_HALF_WIDTH_FACTOR = 0.15  # ±15% at 90% confidence

    def __init__(
        self,
        gpu_count: int = 1,
        calibration_factor: float = 1.0,
    ) -> None:
        if gpu_count < 1:
            raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")
        self._gpu_count = gpu_count
        self._calibration_factor = calibration_factor

    def predict_from_name(
        self,
        model_name: str,
        batch_size: int,
        **kwargs: Any,
    ) -> MemoryEstimate:
        """Predict memory using a model name string instead of param count.

        Args:
            model_name: A recognised model name (e.g. ``"llama-3-8b"``).
            batch_size: Per-device micro-batch size.
            **kwargs: Additional keyword arguments forwarded to :meth:`predict`.

        Returns:
            A :class:`MemoryEstimate`.

        Raises:
            ValueError: If the model name is not in the lookup table.

        Examples:
            >>> model = MemoryModel()
            >>> est = model.predict_from_name("gpt2", batch_size=2)
            >>> est.peak_memory_mb > 0
            True
        """
        param_count = _params_from_name(model_name)
        return self.predict(param_count=param_count, batch_size=batch_size, **kwargs)

    # ------------------------------------------------------------------
    # Sub-components
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_arch(
        param_count: int,
        hidden_dim: Optional[int],
        num_layers: Optional[int],
    ) -> Tuple[int, int]:
        """Infer hidden_dim and num_layers from param count when not provided.

        Only the missing dimension(s) are inferred; explicit values pass
        through unchanged. See :func:`_infer_hidden_layers`.
        """
        inferred_h, inferred_l = _infer_hidden_layers(param_count)
        return (
            hidden_dim if hidden_dim else inferred_h,
            num_layers if num_layers else inferred_l,
        )

    @staticmethod
    def _optimizer_states_mb(
        param_count: int, optimizer: str, parallelism: str
    ) -> float:
        """Compute optimizer-state memory in MiB.

        AdamW/Adam always store two FP32 moments per parameter.
        Adafactor stores a factored representation ≈ 0.5×.
        SGD has no optimizer state (weight-decay is applied inline).
        ZeRO stage 1/2/3 shard the optimizer states across devices.
        """
        fp32_param_mb = param_count * 4.0 / 1024 / 1024  # FP32 copy

        if optimizer in {"adamw", "adam"}:
            opt_mb = 2.0 * fp32_param_mb   # m (fp32) + v (fp32)
            # Master weights in mixed precision training
            opt_mb += fp32_param_mb
        elif optimizer == "adafactor":
            opt_mb = 0.5 * fp32_param_mb
        elif optimizer == "sgd":
            opt_mb = 0.0
        else:
            opt_mb = 2.0 * fp32_param_mb  # conservative default

        # ZeRO shards optimizer states starting at stage 1
        if parallelism in {"zero1", "zero2", "zero3"}:
            # We don't have gpu_count here, but MemoryModel has it as instance attr
            # This is called from within predict() where we use self._gpu_count
            # We'll handle sharding at the caller level; return unshareded here.
            pass

        return opt_mb

    def _optimizer_states_mb_sharded(
        self, param_count: int, optimizer: str, parallelism: str
    ) -> float:
        """Optimizer state memory accounting for ZeRO sharding."""
        opt_mb = self._optimizer_states_mb(param_count, optimizer, parallelism)
        if parallelism in {"zero1", "zero2", "zero3"}:
            opt_mb /= max(self._gpu_count, 1)
        return opt_mb

    @staticmethod
    def _activations_mb(
        batch_size: int,
        sequence_length: int,
        hidden_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool,
        precision: PrecisionMode,
    ) -> float:
        """Estimate activation memory in MiB.

        Uses the standard transformer activation formula:
        bytes ≈ layers × batch × seq_len × hidden × 34  (for standard attn)
        where the factor 34 accounts for all intermediate tensors in a
        transformer block (QKV, attention scores, FFN, etc.).

        Gradient checkpointing reduces activations by sqrt(num_layers) by
        only storing activations at checkpointed layer boundaries.

        Args:
            batch_size: Per-device micro-batch size.
            sequence_length: Token sequence length.
            hidden_dim: Model hidden dimension.
            num_layers: Number of transformer layers.
            use_gradient_checkpointing: If ``True``, apply the sqrt reduction.
            precision: Training precision (determines bytes per element).

        Returns:
            Estimated activation memory in MiB.
        """
        bytes_elem = _BYTES_PER_PARAM[precision]
        # Each layer stores activations for: input, attn projections,
        # attn scores, FFN input/output, residuals ≈ 34 * hidden elements
        act_per_layer = batch_size * sequence_length * hidden_dim * 34 * bytes_elem
        total_act_bytes = act_per_layer * num_layers

        if use_gradient_checkpointing:
            # Checkpointing recomputes activations; only √L checkpoints held
            checkpoint_factor = math.sqrt(num_layers) / num_layers
            total_act_bytes *= checkpoint_factor

        return total_act_bytes / 1024 / 1024

    def predict(
        self,
        param_count: int,
        batch_size: int,
        precision: str = "bf16",
        optimizer: str = "adamw",
        parallelism: str = "none",
        use_gradient_checkpointing: bool = False,
        sequence_length: int = 512,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> MemoryEstimate:
        """Predict peak GPU memory for a single training step.

        Args:
            param_count: Number of model parameters.
            batch_size: Per-device micro-batch size.
            precision: Training precision.  One of
                ``"fp32"``, ``"fp16"``, ``"bf16"``, ``"int8"``, ``"int4"``.
            optimizer: Optimizer type.  One of
                ``"adamw"``, ``"adam"``, ``"sgd"``, ``"adafactor"``.
            parallelism: Parallelism strategy.  One of
                ``"none"``, ``"dp"``, ``"ddp"``, ``"fsdp"``,
                ``"zero1"``, ``"zero2"``, ``"zero3"``.
            use_gradient_checkpointing: Whether gradient checkpointing is enabled.
            sequence_length: Token sequence length (for activation estimate).
            hidden_dim: Model hidden dimension; estimated if not provided.
            num_layers: Number of transformer layers; estimated if not given.

        Returns:
            A :class:`MemoryEstimate` with a central prediction and CI.

        Examples:
            >>> model = MemoryModel()
            >>> est = model.predict(7_000_000_000, 4, "bf16", "adamw")
            >>> est.peak_memory_mb > 10_000
            True
        """
        prec = PrecisionMode(precision.lower())
        bytes_param = _BYTES_PER_PARAM[prec]

        h_dim, n_layers = self._infer_arch(param_count, hidden_dim, num_layers)

        # Parameters
        params_mb = param_count * bytes_param / 1024 / 1024

        # Gradients (ZeRO-2/3/FSDP shards across GPUs)
        grad_mb = param_count * bytes_param / 1024 / 1024
        if parallelism in {"zero2", "zero3", "fsdp"}:
            grad_mb /= max(self._gpu_count, 1)

        # Parameters themselves are sharded in ZeRO-3 / FSDP
        if parallelism in {"zero3", "fsdp"}:
            params_mb /= max(self._gpu_count, 1)

        # Optimizer states
        opt_mb = self._optimizer_states_mb_sharded(param_count, optimizer, parallelism)

        # Activations
        act_mb = self._activations_mb(
            batch_size=batch_size,
            sequence_length=sequence_length,
            hidden_dim=h_dim,
            num_layers=n_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
            precision=prec,
        )

        overhead_mb = 500.0

        breakdown = MemoryBreakdown(
            parameters_mb=params_mb,
            gradients_mb=grad_mb,
            optimizer_states_mb=opt_mb,
            activations_mb=act_mb,
            framework_overhead_mb=overhead_mb,
        )

        raw_total = breakdown.total_mb
        calibrated = raw_total * self._calibration_factor
        half = calibrated * self._CI_HALF_WIDTH_FACTOR

        return MemoryEstimate(
            peak_memory_mb=calibrated,
            lower_mb=max(0.0, calibrated - half),
            upper_mb=calibrated + half,
            breakdown=breakdown,
            gpu_count=self._gpu_count,
            calibration_factor=self._calibration_factor,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, measured_samples: List[dict[str, Any]]) -> float:
        """Fit a linear correction factor from real measurements.

        Uses least squares to find the scalar ``c`` that minimises
        ``Σ (c × predicted_mb - measured_mb)²``.

        Args:
            measured_samples: List of dicts, each with keys matching
                :meth:`predict` parameters PLUS ``"measured_mb"``
                (the actual GPU peak memory observed during a real run).

        Returns:
            The fitted calibration factor (also stored as ``calibration_factor``).

        Raises:
            ValueError: If ``measured_samples`` is empty.

        Examples:
            >>> model = MemoryModel()
            >>> samples = [{"param_count": 125_000_000, "batch_size": 4,
            ...              "measured_mb": 3500.0}]
            >>> factor = model.calibrate(samples)
            >>> isinstance(factor, float)
            True
        """
        if not measured_samples:
            raise ValueError("measured_samples must not be empty")

        predicted_vals: list[float] = []
        measured_vals: list[float] = []

        for sample in measured_samples:
            sample_copy = dict(sample)
            measured_mb = float(sample_copy.pop("measured_mb"))
            pred = self.predict(**sample_copy)
            predicted_vals.append(pred.peak_memory_mb / self._calibration_factor)
            measured_vals.append(measured_mb)

        p = np.array(predicted_vals, dtype=np.float64)
        m = np.array(measured_vals, dtype=np.float64)

        # Least-squares: minimise ||c*p - m||² → c = (p·m) / (p·p)
        factor = float(np.dot(p, m) / np.dot(p, p))
        self._calibration_factor = max(0.1, factor)  # sanity clamp
        return self._calibration_factor

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def calibration_factor(self) -> float:
        """Current calibration factor (1.0 = uncalibrated)."""
        return self._calibration_factor

    @property
    def gpu_count(self) -> int:
        """Number of GPUs this model accounts for."""
        return self._gpu_count
