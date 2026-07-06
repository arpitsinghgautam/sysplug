"""Training stability signal detection.

Monitors a sliding window of (step, loss) tuples for signs of instability:
divergence, oscillation, and gradient norm spikes.  Provides a recommended
corrective action for the online monitor to act on.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Literal, Tuple

Action = Literal["reduce_lr", "increase_grad_clip", "reduce_batch", "ok"]


@dataclass
class StabilityReport:
    """Summary of a stability check over the current window.

    Attributes:
        is_diverging: Loss is trending upward by more than ``diverge_threshold``
            relative to the window minimum.
        is_oscillating: Loss variance over the window exceeds
            ``oscillate_threshold``.
        gradient_norm_spike: A gradient norm spike > 3σ was detected.
        recommended_action: One of ``"reduce_lr"``, ``"increase_grad_clip"``,
            ``"reduce_batch"``, ``"ok"``.
        message: Human-readable explanation of the recommendation.
        window_size: Number of steps in the analysed window.
        current_loss: Most recent loss value.
        loss_trend: Slope of the loss over the window (positive = increasing).
    """

    is_diverging: bool = False
    is_oscillating: bool = False
    gradient_norm_spike: bool = False
    recommended_action: Action = "ok"
    message: str = "Training is stable."
    window_size: int = 0
    current_loss: float = float("nan")
    loss_trend: float = 0.0


class StabilitySignal:
    """Detects training instability from a rolling window of metrics.

    Args:
        window_size: Number of recent steps to analyse.
        diverge_threshold: Fractional increase in loss that triggers a
            divergence warning (default 0.20 = 20% increase from window min).
        oscillate_threshold: Normalised variance threshold for oscillation
            detection (default 0.05).
        grad_norm_sigma: Number of standard deviations above the rolling mean
            at which a gradient norm is flagged as a spike (default 3).

    Examples:
        >>> signal = StabilitySignal(window_size=10)
        >>> for i in range(10):
        ...     signal.record_loss(i, 2.0 - i * 0.1)  # steadily decreasing
        >>> report = signal.check()
        >>> report.recommended_action
        'ok'
    """

    def __init__(
        self,
        window_size: int = 50,
        diverge_threshold: float = 0.20,
        oscillate_threshold: float = 0.05,
        grad_norm_sigma: float = 3.0,
    ) -> None:
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        self._window_size = window_size
        self._diverge_threshold = diverge_threshold
        self._oscillate_threshold = oscillate_threshold
        self._grad_norm_sigma = grad_norm_sigma

        self._losses: Deque[Tuple[int, float]] = deque(maxlen=window_size)
        self._grad_norms: Deque[Tuple[int, float]] = deque(maxlen=window_size)

        # Sticky flag: a non-finite (NaN/Inf) loss is hard divergence and must
        # never be silently dropped. Once seen, the run is flagged as diverged
        # until reset().
        self._nonfinite_loss = False
        self._nonfinite_step = -1

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_loss(self, step: int, loss: float) -> None:
        """Record a loss value at a given training step.

        Args:
            step: Global training step index.
            loss: Loss value. A non-finite (NaN/Inf) loss is recorded as hard
                divergence rather than dropped.
        """
        if math.isfinite(loss):
            self._losses.append((step, loss))
        else:
            self._nonfinite_loss = True
            self._nonfinite_step = step

    def record_grad_norm(self, step: int, grad_norm: float) -> None:
        """Record a gradient L2-norm at a given step.

        Args:
            step: Global training step index.
            grad_norm: Gradient norm (must be non-negative and finite).
        """
        if math.isfinite(grad_norm) and grad_norm >= 0:
            self._grad_norms.append((step, grad_norm))

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def check(self) -> StabilityReport:
        """Analyse the current window and return a stability report.

        Returns:
            A :class:`StabilityReport` with flags, recommended action,
            and a human-readable message.

        Examples:
            >>> signal = StabilitySignal(window_size=5)
            >>> for i in range(5):
            ...     signal.record_loss(i, 1.0 + i * 0.5)  # increasing
            >>> report = signal.check()
            >>> report.is_diverging
            True
        """
        # Hard divergence: a NaN/Inf loss was seen. This outranks everything
        # else and is reported even with an otherwise-empty window.
        if self._nonfinite_loss:
            return StabilityReport(
                is_diverging=True,
                recommended_action="reduce_lr",
                message=(
                    f"Non-finite (NaN/Inf) loss at step {self._nonfinite_step}: "
                    "training has diverged. Reduce the learning rate (and check "
                    "for bad inputs / fp16 overflow)."
                ),
                window_size=len(self._losses),
                current_loss=float("inf"),
            )

        if len(self._losses) < 2:
            return StabilityReport(
                window_size=len(self._losses),
                message="Not enough data yet.",
            )

        losses = [v for _, v in self._losses]
        current_loss = losses[-1]
        window_min = min(losses)

        # 1. Divergence: current loss significantly above window minimum
        is_diverging = False
        if window_min > 0:
            relative_increase = (current_loss - window_min) / window_min
            is_diverging = relative_increase > self._diverge_threshold

        # 2. Oscillation: normalised variance too high
        mean_loss = sum(losses) / len(losses)
        variance = sum((x - mean_loss) ** 2 for x in losses) / len(losses)
        normalised_var = variance / (mean_loss ** 2 + 1e-8)
        is_oscillating = normalised_var > self._oscillate_threshold

        # 3. Gradient norm spike
        grad_norm_spike = self._check_grad_norm_spike()

        # 4. Linear trend (slope via simple linear regression)
        trend = self._compute_trend(losses)

        # 5. Determine recommended action
        action: Action
        message: str
        if is_diverging and grad_norm_spike:
            action = "increase_grad_clip"
            message = (
                "Loss is diverging and gradient norm spikes detected. "
                "Consider increasing gradient clipping or reducing the learning rate."
            )
        elif is_diverging:
            action = "reduce_lr"
            message = (
                f"Loss is diverging (current={current_loss:.4f}, "
                f"window_min={window_min:.4f}). Consider reducing the learning rate."
            )
        elif is_oscillating and grad_norm_spike:
            action = "increase_grad_clip"
            message = "Loss is oscillating with gradient norm spikes. Increase grad_clip."
        elif is_oscillating:
            action = "reduce_batch"
            message = (
                "Loss is oscillating. Consider reducing batch size or learning rate."
            )
        elif grad_norm_spike:
            action = "increase_grad_clip"
            message = "Gradient norm spike detected. Consider tightening grad_clip."
        else:
            action = "ok"
            message = "Training appears stable."

        return StabilityReport(
            is_diverging=is_diverging,
            is_oscillating=is_oscillating,
            gradient_norm_spike=grad_norm_spike,
            recommended_action=action,
            message=message,
            window_size=len(self._losses),
            current_loss=current_loss,
            loss_trend=trend,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_grad_norm_spike(self) -> bool:
        """Return True if the most recent grad norm is a statistical outlier."""
        if len(self._grad_norms) < 3:
            return False
        norms = [v for _, v in self._grad_norms]
        n = len(norms)
        mean = sum(norms) / n
        variance = sum((x - mean) ** 2 for x in norms) / n
        std = math.sqrt(variance + 1e-8)
        latest = norms[-1]
        return latest > mean + self._grad_norm_sigma * std

    @staticmethod
    def _compute_trend(losses: List[float]) -> float:
        """Compute slope of losses via ordinary least squares."""
        n = len(losses)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = (n - 1) / 2.0
        mean_y = sum(losses) / n
        num = sum((xs[i] - mean_x) * (losses[i] - mean_y) for i in range(n))
        denom = sum((xs[i] - mean_x) ** 2 for i in range(n))
        return num / denom if denom != 0 else 0.0

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded history."""
        self._losses.clear()
        self._grad_norms.clear()
        self._nonfinite_loss = False
        self._nonfinite_step = -1

    @property
    def window_size(self) -> int:
        """Configured sliding window size."""
        return self._window_size

    @property
    def num_recorded_steps(self) -> int:
        """Number of loss steps recorded so far."""
        return len(self._losses)
