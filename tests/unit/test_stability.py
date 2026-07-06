"""Unit tests for StabilitySignal."""

from __future__ import annotations

import pytest

from sysplug.stability import StabilitySignal


class TestNonFiniteLoss:
    """A NaN/Inf loss is hard divergence and must be flagged, not dropped."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_loss_flags_divergence(self, bad: float) -> None:
        signal = StabilitySignal(window_size=10)
        signal.record_loss(0, 1.0)
        signal.record_loss(1, bad)
        report = signal.check()
        assert report.is_diverging
        assert report.recommended_action == "reduce_lr"
        assert "diverged" in report.message.lower()

    def test_nonfinite_reported_even_with_empty_window(self) -> None:
        signal = StabilitySignal(window_size=5)
        signal.record_loss(3, float("nan"))  # first and only sample
        report = signal.check()
        assert report.is_diverging
        assert "step 3" in report.message

    def test_reset_clears_divergence(self) -> None:
        signal = StabilitySignal(window_size=5)
        signal.record_loss(0, float("nan"))
        assert signal.check().is_diverging
        signal.reset()
        signal.record_loss(0, 1.0)
        assert not signal.check().is_diverging

    def test_finite_losses_still_recorded(self) -> None:
        signal = StabilitySignal(window_size=5)
        signal.record_loss(0, 1.0)
        signal.record_loss(1, float("nan"))
        signal.record_loss(2, 1.2)
        # NaN did not consume a slot; the two finite losses are both present.
        assert signal.num_recorded_steps == 2


class TestStabilitySignalBasic:
    def test_not_enough_data(self) -> None:
        signal = StabilitySignal(window_size=5)
        signal.record_loss(0, 1.0)
        report = signal.check()
        assert report.recommended_action == "ok"
        assert "Not enough" in report.message

    def test_stable_training(self) -> None:
        signal = StabilitySignal(window_size=10)
        for i in range(10):
            signal.record_loss(i, 2.0 - i * 0.1)  # decreasing loss
        report = signal.check()
        assert report.recommended_action == "ok"
        assert not report.is_diverging
        assert not report.is_oscillating

    def test_diverging_loss(self) -> None:
        signal = StabilitySignal(window_size=10, diverge_threshold=0.10)
        # Start at 1.0, end at 2.0 — 100% increase
        losses = [1.0, 1.1, 1.2, 1.5, 1.8, 2.0, 2.0, 2.0, 2.0, 2.0]
        for i, loss in enumerate(losses):
            signal.record_loss(i, loss)
        report = signal.check()
        assert report.is_diverging
        assert report.recommended_action == "reduce_lr"

    def test_oscillating_loss(self) -> None:
        signal = StabilitySignal(window_size=10, oscillate_threshold=0.01)
        # Alternating high-variance loss
        for i in range(10):
            signal.record_loss(i, 1.0 if i % 2 == 0 else 3.0)
        report = signal.check()
        assert report.is_oscillating

    def test_grad_norm_spike_detection(self) -> None:
        signal = StabilitySignal(window_size=20, grad_norm_sigma=3.0)
        # Normal grad norms around 1.0; also record stable loss so check() proceeds
        for i in range(19):
            signal.record_grad_norm(i, 1.0)
            signal.record_loss(i, 1.0)
        # Spike at last step
        signal.record_grad_norm(19, 100.0)
        signal.record_loss(19, 1.0)
        report = signal.check()
        assert report.gradient_norm_spike

    def test_no_grad_norm_spike_normal(self) -> None:
        signal = StabilitySignal(window_size=10)
        for i in range(10):
            signal.record_grad_norm(i, 1.0 + 0.1 * (i % 3))
        signal.record_loss(0, 1.0)
        signal.record_loss(1, 0.9)
        report = signal.check()
        assert not report.gradient_norm_spike

    def test_diverging_plus_grad_spike_recommends_clip(self) -> None:
        signal = StabilitySignal(window_size=10, diverge_threshold=0.05, grad_norm_sigma=2.0)
        # Both diverging loss and spike
        for i in range(9):
            signal.record_loss(i, 1.0)
            signal.record_grad_norm(i, 1.0)
        signal.record_loss(9, 2.0)
        signal.record_grad_norm(9, 50.0)
        report = signal.check()
        assert report.recommended_action in {"increase_grad_clip", "reduce_lr"}


class TestStabilityReportFields:
    def test_current_loss_field(self) -> None:
        signal = StabilitySignal(window_size=5)
        for i in range(5):
            signal.record_loss(i, float(i + 1))
        report = signal.check()
        assert report.current_loss == pytest.approx(5.0)

    def test_window_size_field(self) -> None:
        signal = StabilitySignal(window_size=20)
        for i in range(10):
            signal.record_loss(i, 1.0)
        report = signal.check()
        assert report.window_size == 10

    def test_loss_trend_positive_for_increasing(self) -> None:
        signal = StabilitySignal(window_size=10)
        for i in range(10):
            signal.record_loss(i, float(i))  # increasing
        report = signal.check()
        assert report.loss_trend > 0

    def test_loss_trend_negative_for_decreasing(self) -> None:
        signal = StabilitySignal(window_size=10)
        for i in range(10):
            signal.record_loss(i, float(10 - i))  # decreasing
        report = signal.check()
        assert report.loss_trend < 0


class TestStabilityReset:
    def test_reset_clears_history(self) -> None:
        signal = StabilitySignal(window_size=5)
        for i in range(5):
            signal.record_loss(i, float(i))
        assert signal.num_recorded_steps == 5
        signal.reset()
        assert signal.num_recorded_steps == 0

    def test_after_reset_not_enough_data(self) -> None:
        signal = StabilitySignal(window_size=5)
        for i in range(5):
            signal.record_loss(i, float(i) * 2.0)
        signal.reset()
        report = signal.check()
        assert "Not enough" in report.message


class TestStabilityEdgeCases:
    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            StabilitySignal(window_size=1)

    def test_nan_loss_not_recorded(self) -> None:
        signal = StabilitySignal(window_size=5)
        signal.record_loss(0, float("nan"))
        signal.record_loss(1, 1.0)
        assert signal.num_recorded_steps == 1  # nan not recorded

    def test_inf_loss_not_recorded(self) -> None:
        signal = StabilitySignal(window_size=5)
        signal.record_loss(0, float("inf"))
        assert signal.num_recorded_steps == 0

    def test_negative_grad_norm_not_recorded(self) -> None:
        signal = StabilitySignal(window_size=5)
        signal.record_grad_norm(0, -1.0)
        signal.record_loss(0, 1.0)
        signal.record_loss(1, 0.9)
        report = signal.check()
        assert not report.gradient_norm_spike
