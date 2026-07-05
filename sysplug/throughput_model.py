"""Throughput prediction for deep learning training.

Implements a roofline model for estimating training throughput (samples/sec
and tokens/sec) and an empirical regression mode that can be calibrated with
actual profiling measurements.

The roofline model computes attainable FLOP/s as:
    attainable = min(compute_peak, bandwidth × arithmetic_intensity)

then derives throughput from:
    throughput = attainable_flops / flops_per_step

References:
    - Williams et al. (2009) "Roofline: An Insightful Visual Performance
      Model for Multicore Architectures". CACM 52(4).
    - Kaplan et al. (2020) "Scaling Laws for Neural Language Models".
      https://arxiv.org/abs/2001.08361  — FLOPs estimate: ~6×params×tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sysplug.memory_model import PrecisionMode

# ---------------------------------------------------------------------------
# GPU spec table: (peak_TFLOPS_bf16, peak_TFLOPS_fp16, peak_TFLOPS_fp32)
# Sources: NVIDIA product pages, measured values.
# ---------------------------------------------------------------------------

@dataclass
class _GPUSpec:
    name_key: str
    tflops_fp32: float
    tflops_fp16: float
    tflops_bf16: float


_GPU_SPECS: list[_GPUSpec] = [
    _GPUSpec("H100",    67.0,  989.0,  989.0),
    _GPUSpec("A100",    19.5,  312.0,  312.0),
    _GPUSpec("A10G",     3.1,  125.0,  125.0),
    _GPUSpec("A10",      3.1,  125.0,  125.0),
    _GPUSpec("A30",      5.2,  165.0,  165.0),
    _GPUSpec("A40",      7.4,  149.7,  149.7),
    _GPUSpec("A6000",    7.4,  149.7,  149.7),
    _GPUSpec("V100",    14.0,  125.0,   14.0),
    _GPUSpec("T4",       8.1,   65.0,    8.1),
    _GPUSpec("P100",     9.3,   18.7,    9.3),
    _GPUSpec("RTX 4090", 82.6,  165.2,   82.6),
    _GPUSpec("RTX 4080", 48.7,   97.4,   48.7),
    _GPUSpec("RTX 3090", 35.6,   35.6,    0.0),
    _GPUSpec("RTX 3080", 29.8,   29.8,    0.0),
    _GPUSpec("L40",       9.1,  181.0,  181.0),
    _GPUSpec("L4",        3.1,  121.0,  121.0),
]

_DEFAULT_SPEC = _GPUSpec("unknown", 10.0, 20.0, 20.0)


def _get_gpu_spec(gpu_name: str) -> _GPUSpec:
    """Return the best matching GPU spec for the given name string."""
    name_lower = gpu_name.lower()
    for spec in _GPU_SPECS:
        if spec.name_key.lower() in name_lower:
            return spec
    return _DEFAULT_SPEC


def _flops_per_step(
    model_params: int,
    batch_size: int,
    sequence_length: int,
) -> float:
    """Estimate total FLOPs for one forward+backward pass.

    Uses the standard estimate from Kaplan et al. (2020):
        FLOPs ≈ 6 × params × seq_len × batch_size

    The factor 6 accounts for: multiply-add in forward (2) × 3 for backward
    (rule of thumb: backward ≈ 2× forward cost).

    Args:
        model_params: Number of model parameters.
        batch_size: Per-device micro-batch size.
        sequence_length: Token sequence length.

    Returns:
        Total floating-point operations as a float.
    """
    return 6.0 * model_params * sequence_length * batch_size


def _peak_tflops(spec: _GPUSpec, precision: PrecisionMode) -> float:
    """Return GPU peak FLOP/s (in TFLOPS) for the given precision."""
    if precision in {PrecisionMode.FP16}:
        return spec.tflops_fp16
    if precision in {PrecisionMode.BF16}:
        return spec.tflops_bf16
    return spec.tflops_fp32


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ThroughputEstimate:
    """Result of :meth:`ThroughputModel.predict`.

    Attributes:
        samples_per_sec: Estimated training throughput in samples/second.
        tokens_per_sec: Estimated throughput in tokens/second.
        flops_per_step: Estimated FLOPs per optimizer step.
        attainable_tflops: Roofline-attainable compute throughput in TFLOPS.
        arithmetic_intensity: Compute-to-memory-traffic ratio.
        is_memory_bound: True when the workload is memory-bandwidth bound.
        gpu_name: GPU used for the estimate.
        calibration_factor: Empirical correction factor (1.0 = uncalibrated).
    """

    samples_per_sec: float
    tokens_per_sec: float
    flops_per_step: float
    attainable_tflops: float
    arithmetic_intensity: float
    is_memory_bound: bool
    gpu_name: str
    calibration_factor: float = 1.0


# ---------------------------------------------------------------------------
# ThroughputModel
# ---------------------------------------------------------------------------


class ThroughputModel:
    """Estimates training throughput using a roofline model.

    Provides both a physics-based roofline mode (default) and an empirical
    regression mode that can be fitted from actual measurements.

    Args:
        gpu_name: GPU model name string (used for spec lookup).
        gpu_count: Number of GPUs; throughput scales approximately linearly.
        bandwidth_gbps: Peak memory bandwidth in GB/s (overrides lookup).
        calibration_factor: Initial empirical correction multiplier.

    Examples:
        >>> model = ThroughputModel(gpu_name="A100")
        >>> est = model.predict(
        ...     effective_batch_size=32,
        ...     model_size_params=7_000_000_000,
        ...     precision="bf16",
        ... )
        >>> est.samples_per_sec > 0
        True
    """

    # Model efficiency — real kernels achieve ~30-55% of peak FLOP/s
    _HW_EFFICIENCY = 0.45

    def __init__(
        self,
        gpu_name: str = "A100",
        gpu_count: int = 1,
        bandwidth_gbps: Optional[float] = None,
        calibration_factor: float = 1.0,
    ) -> None:
        self._gpu_name = gpu_name
        self._gpu_count = max(1, gpu_count)
        self._spec = _get_gpu_spec(gpu_name)
        self._bandwidth_gbps = bandwidth_gbps or self._spec.tflops_fp32 * 20  # heuristic
        # Use hardware bandwidth table if available
        try:
            from sysplug.hardware import _estimate_bandwidth
            self._bandwidth_gbps = bandwidth_gbps or _estimate_bandwidth(gpu_name)
        except ImportError:
            pass
        self._calibration_factor = calibration_factor

        # Empirical regression model (set after calibrate())
        self._empirical_coeffs: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        effective_batch_size: int,
        model_size_params: int,
        precision: str = "bf16",
        sequence_length: int = 512,
    ) -> ThroughputEstimate:
        """Predict training throughput using the roofline model.

        Args:
            effective_batch_size: Total batch size (per-device × grad_acc × gpus).
            model_size_params: Number of model parameters.
            precision: Training precision string.
            sequence_length: Token sequence length.

        Returns:
            A :class:`ThroughputEstimate` with throughput and roofline metrics.

        Raises:
            ValueError: If effective_batch_size or model_size_params ≤ 0.

        Examples:
            >>> m = ThroughputModel(gpu_name="V100")
            >>> est = m.predict(effective_batch_size=16, model_size_params=1_000_000_000)
            >>> est.tokens_per_sec > 0
            True
        """
        if effective_batch_size <= 0:
            raise ValueError(f"effective_batch_size must be > 0, got {effective_batch_size}")
        if model_size_params <= 0:
            raise ValueError(f"model_size_params must be > 0, got {model_size_params}")

        prec = PrecisionMode(precision.lower())

        # If empirical model is available and this is in its range, use it
        if self._empirical_coeffs is not None:
            return self._predict_empirical(
                effective_batch_size, model_size_params, prec, sequence_length
            )

        return self._predict_roofline(
            effective_batch_size, model_size_params, prec, sequence_length
        )

    def _predict_roofline(
        self,
        effective_batch_size: int,
        model_size_params: int,
        prec: PrecisionMode,
        sequence_length: int,
    ) -> ThroughputEstimate:
        """Roofline-based throughput estimate."""
        peak_tflops = _peak_tflops(self._spec, prec) * self._gpu_count
        flops = _flops_per_step(model_size_params, effective_batch_size, sequence_length)

        # Arithmetic intensity: FLOPs / bytes moved through memory
        # For a transformer: ~6×params FLOPs, memory traffic ~2×params bytes
        bytes_traffic = 2.0 * model_size_params * effective_batch_size
        arith_intensity = flops / max(bytes_traffic, 1.0)

        # Attainable FLOP/s from roofline (GFlops/s)
        compute_roof = peak_tflops * 1e12  # TFLOPS → FLOPS
        memory_roof = self._bandwidth_gbps * 1e9 * arith_intensity

        attainable_flops = min(compute_roof, memory_roof) * self._HW_EFFICIENCY
        is_memory_bound = memory_roof < compute_roof

        # Time per step (seconds)
        step_time_sec = flops / max(attainable_flops, 1.0)

        samples_per_sec = (effective_batch_size / step_time_sec) * self._calibration_factor
        tokens_per_sec = samples_per_sec * sequence_length

        return ThroughputEstimate(
            samples_per_sec=max(0.0, samples_per_sec),
            tokens_per_sec=max(0.0, tokens_per_sec),
            flops_per_step=flops,
            attainable_tflops=attainable_flops / 1e12,
            arithmetic_intensity=arith_intensity,
            is_memory_bound=is_memory_bound,
            gpu_name=self._gpu_name,
            calibration_factor=self._calibration_factor,
        )

    def _predict_empirical(
        self,
        effective_batch_size: int,
        model_size_params: int,
        prec: PrecisionMode,
        sequence_length: int,
    ) -> ThroughputEstimate:
        """Empirical regression-based throughput estimate."""
        assert self._empirical_coeffs is not None
        a, b = self._empirical_coeffs
        samples_per_sec = max(0.0, a * effective_batch_size + b)
        samples_per_sec *= self._calibration_factor

        # Compute roofline metrics for the metadata fields
        roofline = self._predict_roofline(
            effective_batch_size, model_size_params, prec, sequence_length
        )

        return ThroughputEstimate(
            samples_per_sec=samples_per_sec,
            tokens_per_sec=samples_per_sec * sequence_length,
            flops_per_step=roofline.flops_per_step,
            attainable_tflops=roofline.attainable_tflops,
            arithmetic_intensity=roofline.arithmetic_intensity,
            is_memory_bound=roofline.is_memory_bound,
            gpu_name=self._gpu_name,
            calibration_factor=self._calibration_factor,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_roofline(self, measured_samples: List[Dict[str, Any]]) -> float:
        """Fit a scalar correction factor against real throughput measurements.

        Args:
            measured_samples: List of dicts with keys ``"effective_batch_size"``,
                ``"model_size_params"``, ``"precision"``, ``"sequence_length"``
                (optional), and ``"measured_samples_per_sec"``.

        Returns:
            The fitted calibration factor.

        Raises:
            ValueError: If ``measured_samples`` is empty.

        Examples:
            >>> m = ThroughputModel(gpu_name="A100")
            >>> samples = [{"effective_batch_size": 32,
            ...              "model_size_params": 125_000_000,
            ...              "precision": "bf16",
            ...              "measured_samples_per_sec": 120.0}]
            >>> factor = m.calibrate_roofline(samples)
            >>> 0 < factor < 10
            True
        """
        if not measured_samples:
            raise ValueError("measured_samples must not be empty")

        predicted_vals: list[float] = []
        measured_vals: list[float] = []

        self._calibration_factor = 1.0  # predict without correction

        for s in measured_samples:
            sc = dict(s)
            measured = float(sc.pop("measured_samples_per_sec"))
            prec = sc.pop("precision", "bf16")
            seq_len = sc.pop("sequence_length", 512)
            est = self._predict_roofline(
                sc["effective_batch_size"],
                sc["model_size_params"],
                PrecisionMode(prec),
                seq_len,
            )
            predicted_vals.append(est.samples_per_sec)
            measured_vals.append(measured)

        p = np.array(predicted_vals, dtype=np.float64)
        m = np.array(measured_vals, dtype=np.float64)
        factor = float(np.dot(p, m) / np.dot(p, p))
        self._calibration_factor = max(0.01, min(10.0, factor))
        return self._calibration_factor

    def fit_empirical(self, measured_samples: List[Dict[str, Any]]) -> None:
        """Fit a linear empirical model: samples_per_sec = a*batch + b.

        After calling this, :meth:`predict` uses the empirical model
        instead of the roofline for interpolation within the observed range.

        Args:
            measured_samples: List of dicts with keys
                ``"effective_batch_size"`` and ``"measured_samples_per_sec"``.

        Raises:
            ValueError: If fewer than 2 samples are provided.
        """
        if len(measured_samples) < 2:
            raise ValueError("Need at least 2 samples to fit an empirical model")

        batches = np.array([s["effective_batch_size"] for s in measured_samples], dtype=float)
        throughputs = np.array(
            [s["measured_samples_per_sec"] for s in measured_samples], dtype=float
        )

        # Fit y = a*x + b via lstsq
        A = np.column_stack([batches, np.ones_like(batches)])
        coeffs, _, _, _ = np.linalg.lstsq(A, throughputs, rcond=None)
        self._empirical_coeffs = (float(coeffs[0]), float(coeffs[1]))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def calibration_factor(self) -> float:
        """Current calibration factor."""
        return self._calibration_factor

    @property
    def gpu_name(self) -> str:
        """GPU name used for roofline estimates."""
        return self._gpu_name
