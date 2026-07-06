"""Learning-rate and warmup scaling rules for batch size changes.

These rules codify well-established heuristics from the deep learning literature
for adjusting the learning rate when the effective batch size changes.

References:
    - Goyal et al. (2017) "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
      https://arxiv.org/abs/1706.02677  — Linear scaling rule.
    - Krizhevsky (2014) "One weird trick for parallelizing convolutional neural networks"
      https://arxiv.org/abs/1404.5997  — Sqrt scaling (gradient noise scale heuristic).
"""

from __future__ import annotations

import math


def linear_lr_scale(base_lr: float, base_batch: int, new_batch: int) -> float:
    """Scale the learning rate linearly with the batch size ratio.

    Implements the linear scaling rule from Goyal et al. (2017):
    ``new_lr = base_lr * (new_batch / base_batch)``.

    This rule is appropriate when the batch size is below the "noise scale"
    threshold (typically 32 × base_batch), where SGD noise dominates.

    Args:
        base_lr: The reference learning rate corresponding to ``base_batch``.
        base_batch: The reference effective batch size.
        new_batch: The new effective batch size.

    Returns:
        The scaled learning rate.

    Raises:
        ValueError: If any argument is non-positive.

    Examples:
        >>> linear_lr_scale(1e-4, 32, 64)
        0.0002
        >>> linear_lr_scale(3e-5, 8, 64)  # 8x increase
        0.00024
    """
    if base_lr <= 0:
        raise ValueError(f"base_lr must be positive, got {base_lr}")
    if base_batch <= 0:
        raise ValueError(f"base_batch must be positive, got {base_batch}")
    if new_batch <= 0:
        raise ValueError(f"new_batch must be positive, got {new_batch}")

    return base_lr * (new_batch / base_batch)


def sqrt_lr_scale(base_lr: float, base_batch: int, new_batch: int) -> float:
    """Scale the learning rate by the square root of the batch size ratio.

    Implements the sqrt scaling rule from Krizhevsky (2014), motivated by
    the gradient noise scale heuristic: ``new_lr = base_lr * sqrt(new_batch / base_batch)``.

    This rule is appropriate when the batch size is large (above the noise
    scale threshold), where curvature/sharpness effects dominate.

    Args:
        base_lr: The reference learning rate corresponding to ``base_batch``.
        base_batch: The reference effective batch size.
        new_batch: The new effective batch size.

    Returns:
        The scaled learning rate.

    Raises:
        ValueError: If any argument is non-positive.

    Examples:
        >>> import math
        >>> sqrt_lr_scale(1e-4, 32, 128)  # 4x batch -> 2x lr
        0.0002
        >>> round(sqrt_lr_scale(3e-5, 8, 32), 8)
        6e-05
    """
    if base_lr <= 0:
        raise ValueError(f"base_lr must be positive, got {base_lr}")
    if base_batch <= 0:
        raise ValueError(f"base_batch must be positive, got {base_batch}")
    if new_batch <= 0:
        raise ValueError(f"new_batch must be positive, got {new_batch}")

    return base_lr * math.sqrt(new_batch / base_batch)


def warmup_steps_for_batch(base_warmup: int, base_batch: int, new_batch: int) -> int:
    """Scale warmup steps proportionally to the batch size ratio.

    When the effective batch size increases, fewer optimizer steps are taken
    per epoch, so the warmup period (in steps) should be scaled accordingly
    to maintain the same warmup duration in terms of training tokens/samples.

    Args:
        base_warmup: The reference number of warmup steps for ``base_batch``.
        base_batch: The reference effective batch size.
        new_batch: The new effective batch size.

    Returns:
        The adjusted number of warmup steps (rounded to nearest integer, min 1).

    Raises:
        ValueError: If any argument is non-positive.

    Examples:
        >>> warmup_steps_for_batch(100, 32, 64)  # 2x batch -> half the steps
        50
        >>> warmup_steps_for_batch(500, 8, 32)
        125
    """
    if base_warmup <= 0:
        raise ValueError(f"base_warmup must be positive, got {base_warmup}")
    if base_batch <= 0:
        raise ValueError(f"base_batch must be positive, got {base_batch}")
    if new_batch <= 0:
        raise ValueError(f"new_batch must be positive, got {new_batch}")

    scaled = base_warmup * (base_batch / new_batch)
    return max(1, round(scaled))


def recommended_lr_rule(training_type: str, batch_size: int) -> str:
    """Recommend whether to use linear or sqrt learning-rate scaling.

    The recommendation is based on the regime:
    - Small batches (< 256): linear scaling is more robust.
    - Large batches (>= 256): sqrt scaling avoids over-shooting.
    - RLHF/PPO: always sqrt (rewards are noisy; aggressive LR increase is destabilizing).

    Args:
        training_type: One of ``"supervised"``, ``"sft"``, ``"dpo"``, ``"rlhf"``,
            ``"grpo"``.
        batch_size: The effective batch size (batch × grad_acc × gpu_count).

    Returns:
        ``"linear"`` or ``"sqrt"``.

    Raises:
        ValueError: If ``training_type`` is not recognized.

    Examples:
        >>> recommended_lr_rule("sft", 32)
        'linear'
        >>> recommended_lr_rule("rlhf", 512)
        'sqrt'
        >>> recommended_lr_rule("supervised", 512)
        'sqrt'
    """
    valid_types = {"supervised", "sft", "dpo", "rlhf", "grpo"}
    if training_type not in valid_types:
        raise ValueError(
            f"training_type must be one of {sorted(valid_types)}, got '{training_type}'"
        )

    # RLHF/DPO/GRPO: always use sqrt due to policy gradient variance
    if training_type in {"rlhf", "grpo"}:
        return "sqrt"

    # Large batch regime: sqrt avoids over-acceleration
    if batch_size >= 256:
        return "sqrt"

    return "linear"
