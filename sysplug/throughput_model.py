"""Throughput prediction for deep learning training.

Implements a roofline model for estimating training throughput (samples/sec
and tokens/sec) plus an empirical calibration mode that can be fitted from
real profiling measurements.

Model
-----
Time for one optimizer step is modelled as a *compute* term plus a fixed
*per-step overhead* (kernel-launch latency, CPU↔GPU sync, non-overlapped
collectives)::

    step_time = flops_per_step / attainable_flops  +  step_overhead

    samples_per_sec = effective_batch_size / step_time

The compute term uses a roofline for ``attainable_flops``::

    attainable = min(peak_compute, bandwidth × arithmetic_intensity) × efficiency

Arithmetic intensity is FLOPs divided by bytes moved through HBM. Crucially,
memory traffic has **two** terms with different batch scaling:

* **Weights** are read once per step and *reused across the whole batch*, so
  weight/gradient traffic is ``O(params)`` — **independent of batch size**.
* **Activations** are written in the forward pass and read in the backward
  pass, so activation traffic is ``O(layers × batch × seq × hidden)`` — it
  **scales with batch size**.

Because of this, arithmetic intensity *grows* with batch size (weight-dominated
and memory-bound at small batch → activation-dominated and compute-bound at
large batch). Combined with the fixed per-step overhead, predicted throughput
follows the real "ramp then plateau" curve rather than being constant.

.. note::
   The uncalibrated roofline is a *prior*. For trustworthy numbers, calibrate
   against real measurements with :meth:`ThroughputModel.fit_empirical`
   (per-config step-time fit) or :meth:`ThroughputModel.calibrate_roofline`
   (scalar correction).

References:
    - Williams et al. (2009) "Roofline: An Insightful Visual Performance
      Model for Multicore Architectures". CACM 52(4).
    - Kaplan et al. (2020) "Scaling Laws for Neural Language Models".
      https://arxiv.org/abs/2001.08361  — FLOPs estimate: ~6×params×tokens.
    - Korthikanti et al. (2022) "Reducing Activation Recomputation in Large
      Transformer Models". https://arxiv.org/abs/2205.05198  — activation
      memory ∝ layers × batch × seq × hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sysplug.memory_model import _BYTES_PER_PARAM, PrecisionMode

# ---------------------------------------------------------------------------
# GPU spec table: peak dense tensor-core TFLOPS per precision.
#
# Values are vendor "dense" (non-sparse) tensor throughput; the gap between
# this marketing peak and real achieved FLOP/s is bridged by _HW_EFFICIENCY.
# fp16 and bf16 both use FP32 accumulation in PyTorch AMP, so they are equal
# for every entry. Sources: NVIDIA datasheets.
# ---------------------------------------------------------------------------

@dataclass
class _GPUSpec:
    name_key: str
    tflops_fp32: float
    tflops_fp16: float
    tflops_bf16: float


_GPU_SPECS: list[_GPUSpec] = [
    # Data-center
    _GPUSpec("H200",     67.0,  989.0,  989.0),
    _GPUSpec("H100",     67.0,  989.0,  989.0),
    _GPUSpec("A100",     19.5,  312.0,  312.0),
    _GPUSpec("A10G",      3.1,  125.0,  125.0),
    _GPUSpec("A10",       3.1,  125.0,  125.0),
    _GPUSpec("A30",       5.2,  165.0,  165.0),
    _GPUSpec("A40",       7.4,  149.7,  149.7),
    _GPUSpec("A6000",     7.4,  149.7,  149.7),
    _GPUSpec("V100",     14.0,  125.0,  125.0),
    _GPUSpec("T4",        8.1,   65.0,   65.0),
    _GPUSpec("P100",      9.3,   18.7,    9.3),  # Pascal: no bf16 tensor cores
    _GPUSpec("L40S",     18.0,  362.0,  362.0),
    _GPUSpec("L40",       9.1,  181.0,  181.0),
    _GPUSpec("L4",        3.1,  121.0,  121.0),
    # Consumer / workstation Ada
    _GPUSpec("RTX 4090", 82.6,  165.2,  165.2),
    _GPUSpec("RTX 4080", 48.7,   97.4,   97.4),
    _GPUSpec("RTX 6000 Ada", 91.1, 182.0, 182.0),
    # Consumer Ampere (bf16 == fp16 with FP32 accumulate; previously wrongly 0.0)
    _GPUSpec("RTX 3090", 35.6,   71.0,   71.0),
    _GPUSpec("RTX 3080", 29.8,   59.5,   59.5),
    # Workstation / laptop Blackwell.
    # NOTE: provisional peak — refine via measurement + calibration on device.
    _GPUSpec("RTX PRO 5000", 60.0, 250.0, 250.0),
    _GPUSpec("RTX 5090",  105.0, 210.0, 210.0),
]

_DEFAULT_SPEC = _GPUSpec("unknown", 10.0, 20.0, 20.0)

# Fallback HBM/GDDR bandwidth (GB/s) when no lookup is available.
_DEFAULT_BANDWIDTH_GBPS = 900.0

# ---------------------------------------------------------------------------
# Memory-traffic and overhead coefficients (calibratable priors)
# ---------------------------------------------------------------------------

# Weight-related HBM traffic per step, as a multiple of the parameter bytes:
# read weights in forward + read weights in backward + write gradients ≈ 3×.
# Batch-independent (weights are reused across the whole micro-batch).
_WEIGHT_TRAFFIC = 3.0

# Activation HBM traffic coefficient per (layers × batch × seq × hidden)
# element: activation tensors written in forward and read in backward.
_ACT_TRAFFIC = 12.0

# Fixed per-step overhead (seconds): kernel-launch latency, host↔device sync,
# non-overlapped collectives. This is the term that makes small-batch
# throughput low (latency-bound) and is the primary target of calibration.
_STEP_OVERHEAD_SEC = 2.0e-3


def _get_gpu_spec(gpu_name: str) -> _GPUSpec:
    """Return the best matching GPU spec for the given name string."""
    name_lower = gpu_name.lower()
    for spec in _GPU_SPECS:
        if spec.name_key.lower() in name_lower:
            return spec
    return _DEFAULT_SPEC


def _bytes_per_elem(prec: PrecisionMode) -> float:
    """Bytes per weight element for the given compute precision."""
    return _BYTES_PER_PARAM[prec]


def _infer_hidden_layers(model_params: int) -> Tuple[int, int]:
    """Infer transformer ``(hidden_size, num_layers)`` from a parameter count.

    Uses the standard decoder relation ``params ≈ 12 · layers · hidden²`` with
    an empirically-anchored aspect ratio ``hidden ≈ 2.1 · params**(1/3)``
    (rounded to a multiple of 128). This reproduces real configs far better
    than a flat heuristic — e.g. 7B → ~3968 hidden / ~37 layers (real Llama-7B
    is 4096 / 32), 70B → ~8704 / ~77 (real 8192 / 80).

    Args:
        model_params: Total parameter count.

    Returns:
        ``(hidden_size, num_layers)``, both ≥ their sensible floors.
    """
    hidden = int(round(2.1 * (model_params ** (1.0 / 3.0)) / 128.0)) * 128
    hidden = max(128, hidden)
    layers = int(round(model_params / (12.0 * hidden * hidden)))
    layers = max(1, layers)
    return hidden, layers


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
    """Return GPU peak FLOP/s (in TFLOPS) for the given precision.

    Falls back from bf16 to fp16 to fp32 if a column is missing/zero so a
    stale spec entry can never zero out predicted throughput.
    """
    if precision is PrecisionMode.FP16:
        candidate = spec.tflops_fp16
    elif precision is PrecisionMode.BF16:
        candidate = spec.tflops_bf16 or spec.tflops_fp16
    else:
        candidate = spec.tflops_fp32
    return candidate or spec.tflops_fp32 or _DEFAULT_SPEC.tflops_fp32


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
        arithmetic_intensity: Compute-to-memory-traffic ratio (FLOPs/byte).
        is_memory_bound: True when the workload is memory-bandwidth bound.
        gpu_name: GPU used for the estimate.
        calibration_factor: Empirical correction factor (1.0 = uncalibrated).
        step_time_sec: Estimated wall-clock time per optimizer step.
    """

    samples_per_sec: float
    tokens_per_sec: float
    flops_per_step: float
    attainable_tflops: float
    arithmetic_intensity: float
    is_memory_bound: bool
    gpu_name: str
    calibration_factor: float = 1.0
    step_time_sec: float = 0.0


# ---------------------------------------------------------------------------
# ThroughputModel
# ---------------------------------------------------------------------------


class ThroughputModel:
    """Estimates training throughput using a roofline model.

    Provides both a physics-based roofline mode (default) and an empirical
    step-time mode that can be fitted from real measurements.

    Args:
        gpu_name: GPU model name string (used for spec lookup).
        gpu_count: Number of GPUs; throughput scales approximately linearly.
        bandwidth_gbps: Peak memory bandwidth in GB/s (overrides lookup).
        calibration_factor: Initial empirical correction multiplier.
        step_overhead_sec: Fixed per-step overhead in seconds (defaults to
            :data:`_STEP_OVERHEAD_SEC`); the dominant driver of low
            small-batch throughput.

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
        step_overhead_sec: float = _STEP_OVERHEAD_SEC,
    ) -> None:
        self._gpu_name = gpu_name
        self._gpu_count = max(1, gpu_count)
        self._spec = _get_gpu_spec(gpu_name)
        self._step_overhead_sec = max(0.0, step_overhead_sec)

        # Resolve memory bandwidth: explicit arg > hardware lookup > default.
        if bandwidth_gbps is not None:
            self._bandwidth_gbps = bandwidth_gbps
        else:
            self._bandwidth_gbps = _DEFAULT_BANDWIDTH_GBPS
            try:
                from sysplug.hardware import _estimate_bandwidth
                self._bandwidth_gbps = _estimate_bandwidth(gpu_name)
            except ImportError:
                pass

        self._calibration_factor = calibration_factor

        # Empirical step-time model (set after fit_empirical()):
        # step_time_sec ≈ slope · effective_batch_size + intercept
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
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> ThroughputEstimate:
        """Predict training throughput using the roofline model.

        Args:
            effective_batch_size: Total batch size (per-device × grad_acc × gpus).
            model_size_params: Number of model parameters.
            precision: Training precision string.
            sequence_length: Token sequence length.
            hidden_size: Transformer hidden dimension. Inferred from the
                parameter count when omitted.
            num_layers: Number of transformer layers. Inferred when omitted.

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

        inferred_h, inferred_l = _infer_hidden_layers(model_size_params)
        hidden = hidden_size if hidden_size and hidden_size > 0 else inferred_h
        layers = num_layers if num_layers and num_layers > 0 else inferred_l

        # If an empirical model is available, use it (roofline still computed
        # for the metadata fields).
        if self._empirical_coeffs is not None:
            return self._predict_empirical(
                effective_batch_size, model_size_params, prec, sequence_length, hidden, layers
            )

        return self._predict_roofline(
            effective_batch_size, model_size_params, prec, sequence_length, hidden, layers
        )

    def _predict_roofline(
        self,
        effective_batch_size: int,
        model_size_params: int,
        prec: PrecisionMode,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
    ) -> ThroughputEstimate:
        """Roofline-based throughput estimate with per-step overhead."""
        peak_tflops = _peak_tflops(self._spec, prec) * self._gpu_count
        flops = _flops_per_step(model_size_params, effective_batch_size, sequence_length)

        # --- Memory traffic (bytes moved through HBM per step) ---------------
        elem_bytes = _bytes_per_elem(prec)
        # Weights: read fwd + read bwd + write grads; reused across the batch
        # → independent of batch size.
        weight_bytes = _WEIGHT_TRAFFIC * model_size_params * elem_bytes
        # Activations: written in forward, read in backward → scales with batch.
        act_bytes = (
            _ACT_TRAFFIC
            * num_layers
            * effective_batch_size
            * sequence_length
            * hidden_size
            * elem_bytes
        )
        bytes_traffic = weight_bytes + act_bytes
        arith_intensity = flops / max(bytes_traffic, 1.0)

        # --- Roofline attainable FLOP/s --------------------------------------
        compute_roof = peak_tflops * 1e12  # TFLOPS → FLOP/s
        memory_roof = self._bandwidth_gbps * 1e9 * arith_intensity

        attainable_flops = min(compute_roof, memory_roof) * self._HW_EFFICIENCY
        is_memory_bound = memory_roof < compute_roof

        # --- Step time = compute + fixed overhead ----------------------------
        compute_time_sec = flops / max(attainable_flops, 1.0)
        step_time_sec = compute_time_sec + self._step_overhead_sec

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
            step_time_sec=step_time_sec,
        )

    def _predict_empirical(
        self,
        effective_batch_size: int,
        model_size_params: int,
        prec: PrecisionMode,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
    ) -> ThroughputEstimate:
        """Empirical step-time regression estimate.

        Uses ``step_time ≈ slope · batch + intercept`` fitted from real
        measurements, giving ``samples_per_sec = batch / step_time`` — a
        saturating curve that ramps at small batch and plateaus at large batch.
        """
        assert self._empirical_coeffs is not None
        slope, intercept = self._empirical_coeffs
        step_time = slope * effective_batch_size + intercept
        step_time = max(step_time, 1e-9)
        samples_per_sec = (effective_batch_size / step_time) * self._calibration_factor

        # Compute roofline metrics for the metadata fields.
        roofline = self._predict_roofline(
            effective_batch_size, model_size_params, prec, sequence_length, hidden_size, num_layers
        )

        return ThroughputEstimate(
            samples_per_sec=max(0.0, samples_per_sec),
            tokens_per_sec=max(0.0, samples_per_sec * sequence_length),
            flops_per_step=roofline.flops_per_step,
            attainable_tflops=roofline.attainable_tflops,
            arithmetic_intensity=roofline.arithmetic_intensity,
            is_memory_bound=roofline.is_memory_bound,
            gpu_name=self._gpu_name,
            calibration_factor=self._calibration_factor,
            step_time_sec=step_time,
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

        saved_factor = self._calibration_factor
        self._calibration_factor = 1.0  # predict without correction

        try:
            for s in measured_samples:
                sc = dict(s)
                measured = float(sc.pop("measured_samples_per_sec"))
                prec = PrecisionMode(str(sc.pop("precision", "bf16")).lower())
                seq_len = int(sc.pop("sequence_length", 512))
                params = int(sc["model_size_params"])
                h, layers = _infer_hidden_layers(params)
                est = self._predict_roofline(
                    int(sc["effective_batch_size"]), params, prec, seq_len, h, layers
                )
                predicted_vals.append(est.samples_per_sec)
                measured_vals.append(measured)
        except Exception:
            self._calibration_factor = saved_factor
            raise

        p = np.array(predicted_vals, dtype=np.float64)
        m = np.array(measured_vals, dtype=np.float64)
        denom = float(np.dot(p, p))
        factor = float(np.dot(p, m) / denom) if denom > 0 else saved_factor
        self._calibration_factor = max(0.01, min(10.0, factor))
        return self._calibration_factor

    def fit_empirical(self, measured_samples: List[Dict[str, Any]]) -> None:
        """Fit an empirical step-time model from real measurements.

        Fits ``step_time = slope · batch + intercept`` (via least squares) so
        that ``samples_per_sec = batch / step_time`` — a saturating throughput
        curve. After calling this, :meth:`predict` uses the empirical model
        instead of the roofline for interpolation within the observed range.

        Args:
            measured_samples: List of dicts with keys
                ``"effective_batch_size"`` and ``"measured_samples_per_sec"``.

        Raises:
            ValueError: If fewer than 2 samples are provided, or a sample has a
                non-positive batch size or throughput.
        """
        if len(measured_samples) < 2:
            raise ValueError("Need at least 2 samples to fit an empirical model")

        batches = np.array(
            [float(s["effective_batch_size"]) for s in measured_samples], dtype=float
        )
        throughputs = np.array(
            [float(s["measured_samples_per_sec"]) for s in measured_samples], dtype=float
        )
        if np.any(batches <= 0) or np.any(throughputs <= 0):
            raise ValueError("batch sizes and throughputs must be positive")

        # Convert to per-step time and fit step_time = slope·batch + intercept.
        step_times = batches / throughputs
        A = np.column_stack([batches, np.ones_like(batches)])
        coeffs, _, _, _ = np.linalg.lstsq(A, step_times, rcond=None)
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
