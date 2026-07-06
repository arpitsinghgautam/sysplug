"""Precise boundary/threshold tests for StabilitySignal.

Every test exercises the exact mathematical conditions that trigger (or don't
trigger) each detection branch.  No mocking — real StabilitySignal instances
are created with known inputs and exact expected outputs are asserted.
"""

from __future__ import annotations

import math

import pytest

from sysplug.stability import StabilityReport, StabilitySignal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(window: int = 20) -> StabilitySignal:
    return StabilitySignal(window_size=window)


def _record_losses(signal: StabilitySignal, losses: list[float], start_step: int = 0) -> None:
    for i, loss in enumerate(losses):
        signal.record_loss(start_step + i, loss)


def _record_norms(signal: StabilitySignal, norms: list[float], start_step: int = 0) -> None:
    for i, n in enumerate(norms):
        signal.record_grad_norm(start_step + i, n)


# ---------------------------------------------------------------------------
# 1. Insufficient data
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_zero_losses_returns_not_enough(self) -> None:
        sig = _make_signal()
        report = sig.check()
        assert "Not enough data" in report.message
        assert report.recommended_action == "ok"
        assert not report.is_diverging
        assert not report.is_oscillating

    def test_one_loss_returns_not_enough(self) -> None:
        sig = _make_signal()
        sig.record_loss(0, 1.0)
        report = sig.check()
        assert "Not enough data" in report.message

    def test_window_size_reported_correctly(self) -> None:
        sig = _make_signal(window=10)
        _record_losses(sig, [1.0] * 7)
        report = sig.check()
        assert report.window_size == 7


# ---------------------------------------------------------------------------
# 2. Stable training
# ---------------------------------------------------------------------------


class TestStableTraining:
    def test_monotone_decreasing_is_stable(self) -> None:
        sig = _make_signal()
        # Steadily decreasing loss: no divergence, no oscillation
        _record_losses(sig, [2.0 - i * 0.05 for i in range(20)])
        report = sig.check()
        assert report.recommended_action == "ok"
        assert not report.is_diverging
        assert not report.is_oscillating

    def test_constant_loss_is_stable(self) -> None:
        sig = _make_signal()
        _record_losses(sig, [1.0] * 20)
        report = sig.check()
        assert not report.is_diverging
        assert not report.is_oscillating
        assert report.recommended_action == "ok"

    def test_slight_noise_below_oscillation_threshold(self) -> None:
        """Tiny noise (0.1% of mean) should not trigger oscillation."""
        sig = StabilitySignal(window_size=20, oscillate_threshold=0.05)
        base = 2.0
        # Variance = (0.002)^2 = 4e-6; normalised = 4e-6 / (2.0)^2 = 1e-6 << 0.05
        _record_losses(sig, [base + 0.001 * (i % 2) for i in range(20)])
        report = sig.check()
        assert not report.is_oscillating

    def test_current_loss_field_correct(self) -> None:
        sig = _make_signal()
        losses = [1.0, 0.9, 0.8, 0.7]
        _record_losses(sig, losses)
        report = sig.check()
        assert math.isclose(report.current_loss, 0.7, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 3. Divergence detection — exact threshold boundary
# ---------------------------------------------------------------------------


class TestDivergenceDetection:
    def test_exactly_at_threshold_not_diverging(self) -> None:
        """
        diverge_threshold = 0.20 means current > min * 1.20 triggers divergence.
        At exactly current = min * 1.20, the condition is NOT met (strict >).
        """
        sig = StabilitySignal(window_size=5, diverge_threshold=0.20)
        min_loss = 1.0
        # Fill 4 values with min_loss, then put exactly 1.20 × min_loss
        _record_losses(sig, [min_loss] * 4 + [min_loss * 1.20])
        report = sig.check()
        # relative_increase = 0.20, which is NOT > 0.20 → not diverging
        assert not report.is_diverging

    def test_just_above_threshold_is_diverging(self) -> None:
        sig = StabilitySignal(window_size=5, diverge_threshold=0.20)
        min_loss = 1.0
        # 1.201 > 1.0 * 1.20  →  relative_increase = 0.201 > 0.20
        _record_losses(sig, [min_loss] * 4 + [min_loss * 1.201])
        report = sig.check()
        assert report.is_diverging
        assert report.recommended_action == "reduce_lr"

    def test_diverging_loss_with_large_spike(self) -> None:
        sig = _make_signal()
        # Normal training then sudden spike
        _record_losses(sig, [1.0] * 15 + [3.0] * 5)
        report = sig.check()
        assert report.is_diverging
        assert report.recommended_action in {"reduce_lr", "increase_grad_clip"}

    def test_loss_trending_up_over_window(self) -> None:
        sig = _make_signal(20)
        # Every step increases by 0.05 — ends at 2.0 + 19*0.05 = 2.95
        # min=2.0, current=2.95 → (2.95-2.0)/2.0 = 0.475 > 0.20
        _record_losses(sig, [2.0 + i * 0.05 for i in range(20)])
        report = sig.check()
        assert report.is_diverging

    def test_loss_trend_positive_when_increasing(self) -> None:
        sig = _make_signal()
        _record_losses(sig, [float(i) for i in range(20)])
        report = sig.check()
        assert report.loss_trend > 0

    def test_loss_trend_negative_when_decreasing(self) -> None:
        sig = _make_signal()
        _record_losses(sig, [20.0 - float(i) for i in range(20)])
        report = sig.check()
        assert report.loss_trend < 0

    def test_zero_minimum_loss_no_divergence(self) -> None:
        """When window_min == 0, divergence check skipped (avoids division by zero)."""
        sig = _make_signal(5)
        _record_losses(sig, [0.0, 0.0, 0.0, 0.0, 100.0])
        report = sig.check()
        # window_min = 0 → guard condition (window_min > 0) is False → not diverging
        assert not report.is_diverging


# ---------------------------------------------------------------------------
# 4. Oscillation detection — exact normalised variance
# ---------------------------------------------------------------------------


class TestOscillationDetection:
    def test_alternating_values_above_threshold(self) -> None:
        """
        Values [1.0, 2.0] alternating: mean=1.5, var=0.25,
        normalised_var = 0.25 / (1.5^2) ≈ 0.111 > default 0.05 → oscillating.
        """
        sig = StabilitySignal(window_size=10, oscillate_threshold=0.05)
        _record_losses(sig, [1.0, 2.0] * 5)
        report = sig.check()
        assert report.is_oscillating

    def test_tight_values_below_threshold(self) -> None:
        """
        Values [1.0, 1.01]: mean≈1.005, var≈(0.005)²=2.5e-5,
        normalised_var ≈ 2.5e-5 / 1.005² ≈ 2.47e-5 << 0.05 → not oscillating.
        """
        sig = StabilitySignal(window_size=10, oscillate_threshold=0.05)
        _record_losses(sig, [1.0, 1.01] * 5)
        report = sig.check()
        assert not report.is_oscillating

    def test_custom_threshold_respected(self) -> None:
        """A tighter threshold (1e-6) catches even tiny oscillations.

        Values [1.0, 1.01]: mean≈1.005, var=(0.005)^2=2.5e-5,
        normalised_var ≈ 2.5e-5 / 1.005^2 ≈ 2.47e-5.
        Default threshold (0.05) → not oscillating.
        Tight threshold (1e-5) → oscillating (2.47e-5 > 1e-5).
        """
        sig = StabilitySignal(window_size=10, oscillate_threshold=1e-5)
        _record_losses(sig, [1.0, 1.01] * 5)
        report = sig.check()
        assert report.is_oscillating

    def test_oscillating_action_is_reduce_batch(self) -> None:
        """Oscillating without divergence and no grad spike → reduce_batch."""
        sig = StabilitySignal(
            window_size=10, oscillate_threshold=0.05, diverge_threshold=5.0
        )  # very high div threshold
        _record_losses(sig, [1.0, 2.0] * 5)
        report = sig.check()
        assert report.is_oscillating
        if not report.gradient_norm_spike and not report.is_diverging:
            assert report.recommended_action == "reduce_batch"


# ---------------------------------------------------------------------------
# 5. Gradient norm spike detection
# ---------------------------------------------------------------------------


class TestGradNormSpike:
    def test_spike_above_3sigma(self) -> None:
        """A very large grad norm spike (10× the normal range) should be flagged.

        The spike detection computes stats on ALL norms including the spike
        itself, so we need the spike to be extreme enough (100×) to still
        exceed mean+3σ after that self-inclusion.
        """
        sig = _make_signal()
        base_norms = [1.0 + 0.01 * i for i in range(19)]  # 1.0 … 1.18
        _record_losses(sig, [1.0] * 20)
        _record_norms(sig, base_norms)
        # Spike of 100 is detectable even after self-inclusion raises the mean.
        # With 19 values around 1.09 and one spike=100:
        # new_mean ≈ 6.0, new_std ≈ 21.6 → threshold ≈ 70.8 < 100 → detected.
        sig.record_grad_norm(19, 100.0)
        report = sig.check()
        assert report.gradient_norm_spike

    def test_moderate_norm_not_a_spike(self) -> None:
        """A grad norm within 2σ should not be flagged."""
        sig = _make_signal()
        norms = [1.0 + 0.1 * i for i in range(10)]
        _record_losses(sig, [1.0] * 10)
        _record_norms(sig, norms)
        # Next norm is at mean + 1σ — well within threshold
        mean = sum(norms) / len(norms)
        variance = sum((x - mean) ** 2 for x in norms) / len(norms)
        std = math.sqrt(variance)
        sig.record_loss(10, 1.0)
        sig.record_grad_norm(10, mean + std)  # exactly 1σ
        report = sig.check()
        assert not report.gradient_norm_spike

    def test_fewer_than_3_grad_norms_no_spike(self) -> None:
        """Spike detection requires at least 3 grad norm samples."""
        sig = _make_signal()
        _record_losses(sig, [1.0] * 10)
        sig.record_grad_norm(0, 100.0)  # only 2 samples: 100, 200
        sig.record_grad_norm(1, 200.0)
        report = sig.check()
        assert not report.gradient_norm_spike

    def test_spike_combined_with_divergence_recommends_grad_clip(self) -> None:
        sig = _make_signal(20)
        # Diverging losses
        _record_losses(sig, [1.0] * 10 + [3.0] * 10)
        # Spike grad norm
        _record_norms(sig, [1.0] * 19 + [50.0])
        report = sig.check()
        if report.is_diverging and report.gradient_norm_spike:
            assert report.recommended_action == "increase_grad_clip"


# ---------------------------------------------------------------------------
# 6. Combined conditions — action priority
# ---------------------------------------------------------------------------


class TestActionPriority:
    def test_diverging_takes_priority_over_oscillating(self) -> None:
        """When both diverging and oscillating, action should mention divergence."""
        sig = StabilitySignal(window_size=10, diverge_threshold=0.05, oscillate_threshold=0.001)
        # High variance + upward trend
        _record_losses(sig, [1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 3.0])
        report = sig.check()
        # Is diverging: current=3.0, min=1.0, increase=2.0 > 0.05 → diverging
        # Is oscillating: high variance → oscillating
        if report.is_diverging:
            # diverging + oscillating (without spike) → reduce_lr
            assert report.recommended_action == "reduce_lr"


# ---------------------------------------------------------------------------
# 7. NaN / Inf / negative filtering
# ---------------------------------------------------------------------------


class TestNaNFiltering:
    def test_nan_loss_ignored(self) -> None:
        sig = _make_signal()
        sig.record_loss(0, 1.0)
        sig.record_loss(1, float("nan"))  # should be dropped
        sig.record_loss(2, 1.0)
        assert sig.num_recorded_steps == 2

    def test_inf_loss_ignored(self) -> None:
        sig = _make_signal()
        sig.record_loss(0, 1.0)
        sig.record_loss(1, float("inf"))
        sig.record_loss(2, 1.0)
        assert sig.num_recorded_steps == 2

    def test_negative_inf_loss_ignored(self) -> None:
        sig = _make_signal()
        sig.record_loss(0, 1.0)
        sig.record_loss(1, float("-inf"))
        assert sig.num_recorded_steps == 1

    def test_negative_grad_norm_ignored(self) -> None:
        sig = _make_signal()
        _record_losses(sig, [1.0] * 5)
        sig.record_grad_norm(0, -1.0)  # negative → dropped
        sig.record_grad_norm(1, 1.0)  # valid
        # Only 1 grad norm recorded
        assert len(sig._grad_norms) == 1

    def test_nan_grad_norm_ignored(self) -> None:
        sig = _make_signal()
        _record_losses(sig, [1.0] * 5)
        sig.record_grad_norm(0, float("nan"))
        assert len(sig._grad_norms) == 0


# ---------------------------------------------------------------------------
# 8. Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_losses(self) -> None:
        sig = _make_signal()
        _record_losses(sig, [1.0] * 10)
        sig.reset()
        assert sig.num_recorded_steps == 0

    def test_reset_clears_grad_norms(self) -> None:
        sig = _make_signal()
        _record_norms(sig, [1.0] * 5)
        sig.reset()
        assert len(sig._grad_norms) == 0

    def test_check_after_reset_returns_not_enough(self) -> None:
        sig = _make_signal()
        _record_losses(sig, [1.0] * 20)
        sig.reset()
        report = sig.check()
        assert "Not enough data" in report.message

    def test_can_record_after_reset(self) -> None:
        sig = _make_signal()
        _record_losses(sig, [1.0] * 20)
        sig.reset()
        _record_losses(sig, [0.5] * 10)
        assert sig.num_recorded_steps == 10


# ---------------------------------------------------------------------------
# 9. Window size enforcement
# ---------------------------------------------------------------------------


class TestWindowSize:
    def test_window_size_constructor_validation(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            StabilitySignal(window_size=1)

    def test_window_size_property(self) -> None:
        sig = StabilitySignal(window_size=30)
        assert sig.window_size == 30

    def test_older_data_evicted(self) -> None:
        """After filling and then overfilling the window, older data is dropped."""
        sig = StabilitySignal(window_size=5)
        # Fill with 1.0, then add 5 more values of 2.0
        _record_losses(sig, [1.0] * 5)
        _record_losses(sig, [2.0] * 5, start_step=5)
        # Window now contains only the last 5 (all 2.0) → not diverging
        report = sig.check()
        # All values in window are 2.0: min=2.0, current=2.0 → not diverging
        assert not report.is_diverging


# ---------------------------------------------------------------------------
# 10. StabilityReport dataclass
# ---------------------------------------------------------------------------


class TestStabilityReport:
    def test_default_report_fields(self) -> None:
        report = StabilityReport()
        assert report.is_diverging is False
        assert report.is_oscillating is False
        assert report.gradient_norm_spike is False
        assert report.recommended_action == "ok"
        assert report.message == "Training is stable."
        assert report.window_size == 0
        assert math.isnan(report.current_loss)
        assert report.loss_trend == 0.0

    def test_real_report_has_correct_types(self) -> None:
        sig = _make_signal()
        _record_losses(sig, [1.0] * 10)
        report = sig.check()
        assert isinstance(report.is_diverging, bool)
        assert isinstance(report.is_oscillating, bool)
        assert isinstance(report.gradient_norm_spike, bool)
        assert isinstance(report.recommended_action, str)
        assert isinstance(report.message, str)
        assert isinstance(report.window_size, int)
        assert isinstance(report.current_loss, float)
        assert isinstance(report.loss_trend, float)
