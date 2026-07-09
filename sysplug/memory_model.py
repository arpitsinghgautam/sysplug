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
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

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

# ---------------------------------------------------------------------------
# Per-family architecture table: (hidden, layers, query_heads, kv_heads).
# Real configs for common families, used when the caller passes a model *name*
# string. Any model passed as an nn.Module is introspected directly from its
# (HuggingFace) config — see resolve_model_arch — so this table only matters
# for name-only inputs. GQA/MQA models have kv_heads < query_heads.
# ---------------------------------------------------------------------------

_MODEL_ARCH_TABLE: dict[str, tuple[int, int, int, int]] = {
    "gpt2-medium": (1024, 24, 16, 16),
    "gpt2-large": (1280, 36, 20, 20),
    "gpt2-xl": (1600, 48, 25, 25),
    "gpt2": (768, 12, 12, 12),
    "llama-2-7b": (4096, 32, 32, 32),
    "llama-2-13b": (5120, 40, 40, 40),
    "llama-2-70b": (8192, 80, 64, 8),
    "llama-3-8b": (4096, 32, 32, 8),
    "llama-3-70b": (8192, 80, 64, 8),
    "mistral-7b": (4096, 32, 32, 8),
    "codellama-7b": (4096, 32, 32, 32),
    "codellama-13b": (5120, 40, 40, 40),
    "qwen-7b": (4096, 32, 32, 32),
    "qwen-14b": (5120, 40, 40, 40),
    "gemma-2b": (2048, 18, 8, 1),
    "gemma-7b": (3072, 28, 16, 16),
    "phi-2": (2560, 32, 32, 32),
    "opt-1.3b": (2048, 24, 32, 32),
    "opt-6.7b": (4096, 32, 32, 32),
    "bert-base": (768, 12, 12, 12),
    "bert-large": (1024, 24, 16, 16),
    "t5-base": (768, 12, 12, 12),
    "t5-large": (1024, 24, 16, 16),
}

# Attention implementations that do NOT materialise the full B·S·S scores
# matrix (memory is O(S) rather than O(S^2)).
_SUBQUADRATIC_ATTN = ("flash", "sdpa", "mem_eff", "memory_efficient", "xformers")

# Activation-memory coefficients (per layer). See paper/experiments/calibrate_memory.py.
# The linear coefficient is calibrated against measured training peaks (~54); the
# theoretical stored-activation count is ~34 (Korthikanti et al. 2022), but the real
# peak also holds autograd bookkeeping and attention-kernel workspace.
_ACT_LINEAR_COEF = 54.0  # linear activations: QKV proj, FFN, residuals, norms (~C·B·S·H)
# eager attention: B·heads·S·S scores + softmax/dropout buffer. Physics-based;
# validated against measured data requires an eager (non-SDPA) run (deferred).
_ATTN_SCORES_COEF = 2.0


def _is_subquadratic_attn(attn_impl: str | None) -> bool:
    """True if the attention impl avoids the ``O(S^2)`` scores matrix."""
    impl = (attn_impl or "eager").lower()
    return any(key in impl for key in _SUBQUADRATIC_ATTN)


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


def _infer_hidden_layers(param_count: int) -> tuple[int, int]:
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
# Architecture resolution (introspect real models instead of guessing)
# ---------------------------------------------------------------------------


@dataclass
class ModelArch:
    """Resolved transformer architecture used by the memory/throughput models.

    Produced by :func:`resolve_model_arch` from an ``nn.Module`` (introspecting
    a HuggingFace ``config`` when present), a model-name string, or a raw
    parameter count. ``source`` records how it was obtained (``"config"`` |
    ``"name_table"`` | ``"param_inference"``).

    Attributes:
        hidden_size: Model hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (< num_heads for GQA/MQA).
        attn_impl: Attention implementation string (e.g. ``"eager"``, ``"sdpa"``,
            ``"flash_attention_2"``).
        max_seq_len: Maximum supported sequence length, if known.
        param_count: Total parameter count.
        source: How the architecture was resolved.
    """

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    attn_impl: str = "eager"
    max_seq_len: int | None = None
    param_count: int = 0
    source: str = "param_inference"

    def is_subquadratic_attn(self) -> bool:
        """True if the attention impl avoids the full ``B·S·S`` scores matrix.

        FlashAttention / SDPA / memory-efficient kernels compute attention in
        tiles and never materialise the ``O(S^2)`` scores, so their activation
        memory scales as ``O(S)`` rather than ``O(S^2)``.
        """
        return _is_subquadratic_attn(self.attn_impl)


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    """Return the first non-None attribute among ``names`` (guarded getattr)."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _safe_param_count(model: Any) -> int:
    """Sum ``model.parameters()``; fall back to 125M with a warning."""
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        warnings.warn(
            f"Could not determine parameter count from model type {type(model)}. "
            "Defaulting to 125M parameters.",
            stacklevel=2,
        )
        return 125_000_000


def _arch_from_name(name: str) -> tuple[int, int, int, int, int]:
    """Return ``(param_count, hidden, layers, heads, kv_heads)`` for a name.

    Uses :data:`_MODEL_ARCH_TABLE` (longest match first, so ``gpt2-medium`` does
    not match ``gpt2``) when the family is known; otherwise infers dims from the
    name's parameter count. Raises ``ValueError`` for a genuinely unknown name
    (same contract as :func:`_params_from_name`).
    """
    key = name.lower().strip()
    params = _params_from_name(name)
    for name_key in sorted(_MODEL_ARCH_TABLE, key=len, reverse=True):
        if name_key in key or key in name_key:
            h, layers, heads, kv = _MODEL_ARCH_TABLE[name_key]
            return params, h, layers, heads, kv
    h, layers = _infer_hidden_layers(params)
    heads = max(1, h // 128)
    return params, h, layers, heads, heads


def resolve_model_arch(model: Any) -> ModelArch:
    """Resolve a :class:`ModelArch` from a model object, name, or param count.

    - ``int`` -> treated as a parameter count; dims inferred.
    - ``str`` -> looked up in the per-family arch table (or inferred from the
      name's parameter count). Raises ``ValueError`` for an unknown name.
    - ``nn.Module`` with a HuggingFace ``.config`` -> dims read directly from
      the config (hidden size, layers, query/KV heads, attention impl).
    - other ``nn.Module`` -> parameter count summed; dims inferred.

    Never raises for a module or int input.

    Examples:
        >>> arch = resolve_model_arch("llama-3-8b")
        >>> (arch.hidden_size, arch.num_layers, arch.num_kv_heads)
        (4096, 32, 8)
    """
    if isinstance(model, bool):  # bool is an int subclass; treat as unknown
        model = 125_000_000
    if isinstance(model, int):
        hidden, layers = _infer_hidden_layers(model)
        heads = max(1, hidden // 128)
        return ModelArch(
            hidden, layers, heads, heads, param_count=int(model), source="param_inference"
        )
    if isinstance(model, str):
        params, hidden, layers, heads, kv = _arch_from_name(model)
        return ModelArch(hidden, layers, heads, kv, param_count=params, source="name_table")

    # Assume an nn.Module-like object.
    param_count = _safe_param_count(model)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        hidden = _first_attr(cfg, ("hidden_size", "n_embd", "d_model", "hidden_dim"))
        layers = _first_attr(cfg, ("num_hidden_layers", "n_layer", "num_layers", "n_layers"))
        heads = _first_attr(cfg, ("num_attention_heads", "n_head", "num_heads"))
        kv = _first_attr(cfg, ("num_key_value_heads", "num_kv_heads"))
        max_seq = _first_attr(cfg, ("max_position_embeddings", "n_positions", "max_seq_len"))
        attn = _first_attr(cfg, ("_attn_implementation", "attn_implementation"))
        if hidden and layers:
            heads = int(heads) if heads else max(1, int(hidden) // 128)
            return ModelArch(
                hidden_size=int(hidden),
                num_layers=int(layers),
                num_heads=heads,
                num_kv_heads=int(kv) if kv else heads,
                attn_impl=str(attn) if attn else "eager",
                max_seq_len=int(max_seq) if max_seq else None,
                param_count=param_count,
                source="config",
            )

    hidden, layers = _infer_hidden_layers(param_count)
    heads = max(1, hidden // 128)
    return ModelArch(
        hidden, layers, heads, heads, param_count=param_count, source="param_inference"
    )


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

    def to_dict(self) -> dict[str, float]:
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
        lower_mb: Lower bound of the confidence band.
        upper_mb: Conservative upper bound; the solver uses this for OOM-safe
            feasibility ("if it fits, it fits").
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

    # Asymmetric confidence band. The upper margin is deliberately wide so the
    # upper bound covers the residual (mostly upward) prediction error — the
    # solver uses upper_mb for OOM-safety. Provisional; re-derived from measured
    # residuals during calibration (see paper/experiments/measure_gpu.py).
    _CI_LOWER_FRAC = 0.10  # lower = peak * (1 - 0.10)
    _CI_UPPER_FRAC = 0.40  # upper = peak * (1 + 0.40); covers measured residuals

    def __init__(
        self,
        gpu_count: int = 1,
        calibration_factor: float = 1.0,
        ci_lower_frac: float | None = None,
        ci_upper_frac: float | None = None,
    ) -> None:
        if gpu_count < 1:
            raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")
        self._gpu_count = gpu_count
        self._calibration_factor = calibration_factor
        self._ci_lower_frac = self._CI_LOWER_FRAC if ci_lower_frac is None else ci_lower_frac
        self._ci_upper_frac = self._CI_UPPER_FRAC if ci_upper_frac is None else ci_upper_frac

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
        hidden_dim: int | None,
        num_layers: int | None,
    ) -> tuple[int, int]:
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
    def _optimizer_states_mb(param_count: int, optimizer: str, parallelism: str) -> float:
        """Compute optimizer-state memory in MiB.

        AdamW/Adam always store two FP32 moments per parameter.
        Adafactor stores a factored representation ≈ 0.5×.
        SGD has no optimizer state (weight-decay is applied inline).
        ZeRO stage 1/2/3 shard the optimizer states across devices.
        """
        fp32_param_mb = param_count * 4.0 / 1024 / 1024  # FP32 copy

        if optimizer in {"adamw", "adam"}:
            opt_mb = 2.0 * fp32_param_mb  # m (fp32) + v (fp32)
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
        num_heads: int,
        attn_scores_materialized: bool,
        use_gradient_checkpointing: bool,
        precision: PrecisionMode,
    ) -> float:
        """Estimate activation memory in MiB (attention-aware).

        Two per-layer terms:

        * **Linear**: QKV projections, FFN, residuals, LayerNorms
          ``≈ 34 · B · S · H`` (Korthikanti et al. 2022).
        * **Attention scores** (eager/full attention only): the materialised
          ``B · heads · S · S`` scores matrix plus its softmax/dropout buffer,
          ``≈ _ATTN_SCORES_COEF · heads · B · S²``. This ``O(S²)`` term
          dominates activation memory at long sequence / large batch.
          FlashAttention / SDPA / memory-efficient kernels never materialise it
          (``attn_scores_materialized=False``), so the term is dropped.

        Gradient checkpointing recomputes activations, keeping only ``√L``
        checkpoints; the ``√L/L`` reduction applies to the whole per-layer term
        (the attention scores are the primary recompute target).

        Args:
            batch_size: Per-device micro-batch size.
            sequence_length: Token sequence length.
            hidden_dim: Model hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of query attention heads (drives the scores term).
            attn_scores_materialized: Whether the full ``S²`` scores are held in
                memory (True for eager, False for Flash/SDPA/mem-efficient).
            use_gradient_checkpointing: If ``True``, apply the ``√L/L`` reduction.
            precision: Training precision (determines bytes per element).

        Returns:
            Estimated activation memory in MiB.
        """
        bytes_elem = _BYTES_PER_PARAM[precision]
        linear = _ACT_LINEAR_COEF * batch_size * sequence_length * hidden_dim
        scores = 0.0
        if attn_scores_materialized:
            scores = _ATTN_SCORES_COEF * num_heads * batch_size * sequence_length * sequence_length
        total_act_bytes = (linear + scores) * num_layers * bytes_elem

        if use_gradient_checkpointing:
            # Checkpointing recomputes activations; only √L checkpoints held.
            total_act_bytes *= math.sqrt(num_layers) / num_layers

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
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        attn_impl: str | None = None,
        arch: ModelArch | None = None,
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
            num_heads: Query attention heads; drives the O(S^2) attention term.
                Estimated if not given.
            attn_impl: Attention implementation. When it is "flash"/"sdpa"/
                memory-efficient the O(S^2) scores term is dropped; defaults to
                "eager" (conservative — keeps the term).
            arch: A :class:`ModelArch` to source the above from. Explicit
                ``hidden_dim``/``num_layers``/``num_heads``/``attn_impl`` kwargs
                still override the corresponding ``arch`` fields.

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

        # Prefer an explicit ModelArch; explicit dim/head/impl kwargs override it.
        if arch is not None:
            hidden_dim = hidden_dim or arch.hidden_size
            num_layers = num_layers or arch.num_layers
            num_heads = num_heads or arch.num_heads
            if attn_impl is None:
                attn_impl = arch.attn_impl

        h_dim, n_layers = self._infer_arch(param_count, hidden_dim, num_layers)
        heads = num_heads if num_heads else max(1, h_dim // 128)
        scores_materialized = not _is_subquadratic_attn(attn_impl)

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
            num_heads=heads,
            attn_scores_materialized=scores_materialized,
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

        return MemoryEstimate(
            peak_memory_mb=calibrated,
            lower_mb=max(0.0, calibrated * (1.0 - self._ci_lower_frac)),
            upper_mb=calibrated * (1.0 + self._ci_upper_frac),
            breakdown=breakdown,
            gpu_count=self._gpu_count,
            calibration_factor=self._calibration_factor,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, measured_samples: list[dict[str, Any]]) -> float:
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
