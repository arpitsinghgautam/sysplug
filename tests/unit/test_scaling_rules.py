"""Unit tests for scaling_rules utilities."""

from __future__ import annotations

import math
import pytest

from sysplug.utils.scaling_rules import (
    linear_lr_scale,
    recommended_lr_rule,
    sqrt_lr_scale,
    warmup_steps_for_batch,
)


class TestLinearLRScale:
    def test_doubles_lr_with_doubled_batch(self) -> None:
        lr = linear_lr_scale(1e-4, base_batch=32, new_batch=64)
        assert lr == pytest.approx(2e-4, rel=1e-6)

    def test_halves_lr_with_halved_batch(self) -> None:
        lr = linear_lr_scale(2e-4, base_batch=64, new_batch=32)
        assert lr == pytest.approx(1e-4, rel=1e-6)

    def test_no_change_same_batch(self) -> None:
        lr = linear_lr_scale(3e-5, 16, 16)
        assert lr == pytest.approx(3e-5, rel=1e-6)

    def test_8x_batch_increase(self) -> None:
        lr = linear_lr_scale(3e-5, 8, 64)
        assert lr == pytest.approx(3e-5 * 8, rel=1e-6)

    def test_invalid_base_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="base_lr"):
            linear_lr_scale(-1e-4, 32, 64)

    def test_invalid_base_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="base_batch"):
            linear_lr_scale(1e-4, 0, 64)

    def test_invalid_new_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="new_batch"):
            linear_lr_scale(1e-4, 32, -1)


class TestSqrtLRScale:
    def test_4x_batch_doubles_lr(self) -> None:
        """Batch × 4 → LR × 2 (sqrt rule)."""
        lr = sqrt_lr_scale(1e-4, base_batch=32, new_batch=128)
        assert lr == pytest.approx(2e-4, rel=1e-4)

    def test_correct_ratio(self) -> None:
        import math
        lr = sqrt_lr_scale(1e-4, 32, 64)
        expected = 1e-4 * math.sqrt(64 / 32)
        assert lr == pytest.approx(expected, rel=1e-6)

    def test_same_batch_no_change(self) -> None:
        lr = sqrt_lr_scale(5e-5, 8, 8)
        assert lr == pytest.approx(5e-5, rel=1e-6)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            sqrt_lr_scale(0.0, 32, 64)
        with pytest.raises(ValueError):
            sqrt_lr_scale(1e-4, 0, 64)

    def test_sqrt_always_less_than_linear(self) -> None:
        """For batch increase > 1, sqrt scaling is more conservative than linear."""
        linear = linear_lr_scale(1e-4, 16, 64)
        sqrt = sqrt_lr_scale(1e-4, 16, 64)
        assert sqrt < linear

    def test_9b_batch_gives_3x(self) -> None:
        lr = sqrt_lr_scale(3e-5, 8, 72)
        expected = 3e-5 * math.sqrt(72 / 8)
        assert lr == pytest.approx(expected, rel=1e-6)


class TestWarmupSteps:
    def test_halved_when_batch_doubled(self) -> None:
        steps = warmup_steps_for_batch(100, 32, 64)
        assert steps == 50

    def test_doubled_when_batch_halved(self) -> None:
        steps = warmup_steps_for_batch(100, 64, 32)
        assert steps == 200

    def test_scales_proportionally(self) -> None:
        steps = warmup_steps_for_batch(500, 8, 32)
        assert steps == 125

    def test_minimum_one(self) -> None:
        # Very large new batch → very small warmup, min=1
        steps = warmup_steps_for_batch(1, 1, 10_000_000)
        assert steps >= 1

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            warmup_steps_for_batch(0, 32, 64)
        with pytest.raises(ValueError):
            warmup_steps_for_batch(100, 0, 64)


class TestRecommendedLRRule:
    def test_sft_small_batch_linear(self) -> None:
        assert recommended_lr_rule("sft", 32) == "linear"

    def test_sft_large_batch_sqrt(self) -> None:
        assert recommended_lr_rule("sft", 512) == "sqrt"

    def test_rlhf_always_sqrt(self) -> None:
        assert recommended_lr_rule("rlhf", 16) == "sqrt"
        assert recommended_lr_rule("rlhf", 1024) == "sqrt"

    def test_grpo_always_sqrt(self) -> None:
        assert recommended_lr_rule("grpo", 32) == "sqrt"

    def test_dpo_small_batch_linear(self) -> None:
        assert recommended_lr_rule("dpo", 64) == "linear"

    def test_supervised_small_batch_linear(self) -> None:
        assert recommended_lr_rule("supervised", 128) == "linear"

    def test_threshold_at_256(self) -> None:
        assert recommended_lr_rule("supervised", 255) == "linear"
        assert recommended_lr_rule("supervised", 256) == "sqrt"

    def test_invalid_training_type_raises(self) -> None:
        with pytest.raises(ValueError, match="training_type"):
            recommended_lr_rule("unknown_type", 32)
