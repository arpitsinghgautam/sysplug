"""Unit tests for MemoryModel."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sysplug.memory_model import MemoryModel, PrecisionMode, _params_from_name


# ---------------------------------------------------------------------------
# Basic bytes-per-param tests
# ---------------------------------------------------------------------------

class TestParameterBytes:
    def test_fp32_7b_parameters(self) -> None:
        """7B params × 4 bytes = 28000 MB."""
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            param_count=7_000_000_000,
            batch_size=1,
            precision="fp32",
            optimizer="sgd",  # no optimizer states
            parallelism="none",
        )
        # Parameters only: 7e9 * 4 / 1024 / 1024 ≈ 26703 MB
        # With overhead and calibration ~15% CI
        assert est.breakdown.parameters_mb == pytest.approx(
            7_000_000_000 * 4 / 1024 / 1024, rel=0.01
        )

    def test_fp16_7b_parameters(self) -> None:
        """7B params × 2 bytes = 14000 MB."""
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            param_count=7_000_000_000,
            batch_size=1,
            precision="fp16",
            optimizer="sgd",
            parallelism="none",
        )
        assert est.breakdown.parameters_mb == pytest.approx(
            7_000_000_000 * 2 / 1024 / 1024, rel=0.01
        )

    def test_bf16_parameters(self) -> None:
        """BF16 and FP16 have the same bytes per param."""
        model = MemoryModel()
        fp16 = model.predict(1_000_000_000, 1, "fp16", "sgd").breakdown.parameters_mb
        bf16 = model.predict(1_000_000_000, 1, "bf16", "sgd").breakdown.parameters_mb
        assert fp16 == pytest.approx(bf16, rel=1e-6)

    def test_int8_half_fp16(self) -> None:
        """INT8 uses half the bytes of FP16."""
        model = MemoryModel()
        fp16 = model.predict(1_000_000_000, 1, "fp16", "sgd").breakdown.parameters_mb
        int8 = model.predict(1_000_000_000, 1, "int8", "sgd").breakdown.parameters_mb
        assert fp16 == pytest.approx(2 * int8, rel=0.01)

    def test_int4_quarter_fp16(self) -> None:
        """INT4 uses a quarter the bytes of FP16."""
        model = MemoryModel()
        fp16 = model.predict(1_000_000_000, 1, "fp16", "sgd").breakdown.parameters_mb
        int4 = model.predict(1_000_000_000, 1, "int4", "sgd").breakdown.parameters_mb
        assert fp16 == pytest.approx(4 * int4, rel=0.01)


# ---------------------------------------------------------------------------
# Optimizer states
# ---------------------------------------------------------------------------

class TestOptimizerStates:
    def test_adamw_two_fp32_moments(self) -> None:
        """AdamW requires 2× FP32 param size for m and v, plus master weights."""
        model = MemoryModel()
        est = model.predict(
            param_count=1_000_000_000,
            batch_size=1,
            precision="bf16",
            optimizer="adamw",
            parallelism="none",
        )
        # BF16 params: 2 bytes, FP32 for optimizer: 3x (m + v + master weight)
        fp32_param_mb = 1_000_000_000 * 4 / 1024 / 1024
        expected_opt_mb = 3 * fp32_param_mb  # 2 moments + master weight
        assert est.breakdown.optimizer_states_mb == pytest.approx(expected_opt_mb, rel=0.01)

    def test_sgd_no_optimizer_states(self) -> None:
        """SGD has no optimizer state tensors."""
        model = MemoryModel()
        est = model.predict(
            param_count=1_000_000_000,
            batch_size=1,
            precision="fp32",
            optimizer="sgd",
            parallelism="none",
        )
        assert est.breakdown.optimizer_states_mb == pytest.approx(0.0, abs=1e-6)

    def test_adafactor_half_fp32(self) -> None:
        """Adafactor states ≈ 0.5× FP32 param size."""
        model = MemoryModel()
        est = model.predict(
            param_count=1_000_000_000,
            batch_size=1,
            precision="bf16",
            optimizer="adafactor",
            parallelism="none",
        )
        fp32_mb = 1_000_000_000 * 4 / 1024 / 1024
        assert est.breakdown.optimizer_states_mb == pytest.approx(0.5 * fp32_mb, rel=0.01)


# ---------------------------------------------------------------------------
# ZeRO sharding
# ---------------------------------------------------------------------------

class TestZeROSharding:
    def test_zero3_shards_everything(self) -> None:
        """ZeRO-3 divides params + grads + optimizer states by gpu_count."""
        gpu_count = 4
        model_no_shard = MemoryModel(gpu_count=1)
        model_zero3 = MemoryModel(gpu_count=gpu_count)

        no_shard = model_no_shard.predict(
            1_000_000_000, 1, "bf16", "adamw", "none"
        )
        with_zero3 = model_zero3.predict(
            1_000_000_000, 1, "bf16", "adamw", "zero3"
        )

        # Parameters and gradients should be sharded
        assert with_zero3.breakdown.parameters_mb < no_shard.breakdown.parameters_mb
        assert with_zero3.breakdown.optimizer_states_mb < no_shard.breakdown.optimizer_states_mb

    def test_zero1_shards_only_optimizer(self) -> None:
        """ZeRO-1 shards optimizer states but not params or grads."""
        model = MemoryModel(gpu_count=4)
        single = MemoryModel(gpu_count=1)

        z1 = model.predict(1_000_000_000, 1, "bf16", "adamw", "zero1")
        no_shard = single.predict(1_000_000_000, 1, "bf16", "adamw", "none")

        # Params should be the same (not sharded under ZeRO-1)
        assert z1.breakdown.parameters_mb == pytest.approx(
            no_shard.breakdown.parameters_mb, rel=0.01
        )
        # Optimizer states should be sharded
        assert z1.breakdown.optimizer_states_mb < no_shard.breakdown.optimizer_states_mb

    def test_zero2_shards_optimizer_and_grads(self) -> None:
        """ZeRO-2 shards optimizer states and gradients."""
        model = MemoryModel(gpu_count=4)
        single = MemoryModel(gpu_count=1)

        z2 = model.predict(1_000_000_000, 1, "bf16", "adamw", "zero2")
        no_shard = single.predict(1_000_000_000, 1, "bf16", "adamw", "none")

        assert z2.breakdown.gradients_mb < no_shard.breakdown.gradients_mb
        assert z2.breakdown.optimizer_states_mb < no_shard.breakdown.optimizer_states_mb


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

class TestGradientCheckpointing:
    def test_checkpointing_reduces_activations(self) -> None:
        """Gradient checkpointing reduces activation memory below full."""
        model = MemoryModel()
        full = model.predict(
            1_000_000_000, 4, "bf16", "adamw", "none",
            use_gradient_checkpointing=False, num_layers=24
        )
        ckpt = model.predict(
            1_000_000_000, 4, "bf16", "adamw", "none",
            use_gradient_checkpointing=True, num_layers=24
        )
        assert ckpt.breakdown.activations_mb < full.breakdown.activations_mb

    def test_checkpointing_sqrt_factor(self) -> None:
        """Activation reduction is approximately sqrt(num_layers) / num_layers."""
        import math
        model = MemoryModel()
        num_layers = 24
        full = model.predict(
            1_000_000_000, 4, "bf16", "sgd", "none",
            use_gradient_checkpointing=False,
            num_layers=num_layers, hidden_dim=1024
        )
        ckpt = model.predict(
            1_000_000_000, 4, "bf16", "sgd", "none",
            use_gradient_checkpointing=True,
            num_layers=num_layers, hidden_dim=1024
        )
        expected_factor = math.sqrt(num_layers) / num_layers
        actual_factor = ckpt.breakdown.activations_mb / full.breakdown.activations_mb
        assert actual_factor == pytest.approx(expected_factor, rel=0.05)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibration_fits_factor(self) -> None:
        """Calibration adjusts predictions toward measured values."""
        model = MemoryModel()
        pred = model.predict(125_000_000, 4)
        # Pretend actual measurement is 80% of prediction
        measured_mb = pred.peak_memory_mb * 0.8

        factor = model.calibrate([{
            "param_count": 125_000_000,
            "batch_size": 4,
            "measured_mb": measured_mb,
        }])

        assert factor == pytest.approx(0.8, rel=0.05)
        # After calibration, prediction should be closer to measured
        recal = model.predict(125_000_000, 4)
        assert recal.peak_memory_mb == pytest.approx(measured_mb, rel=0.1)

    def test_calibration_empty_raises(self) -> None:
        model = MemoryModel()
        with pytest.raises(ValueError, match="must not be empty"):
            model.calibrate([])

    def test_calibration_multiple_samples(self) -> None:
        """Calibration works with multiple samples."""
        model = MemoryModel()
        samples = []
        for bs in [2, 4, 8]:
            pred = model.predict(125_000_000, bs)
            samples.append({
                "param_count": 125_000_000,
                "batch_size": bs,
                "measured_mb": pred.peak_memory_mb * 0.9,
            })
        factor = model.calibrate(samples)
        assert 0.5 < factor < 1.5


# ---------------------------------------------------------------------------
# Model name lookup
# ---------------------------------------------------------------------------

class TestModelNameLookup:
    def test_known_model_name(self) -> None:
        assert _params_from_name("llama-3-8b") == pytest.approx(8_000_000_000, rel=0.01)

    def test_gpt2_name(self) -> None:
        assert _params_from_name("gpt2") == pytest.approx(117_000_000, rel=0.01)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model name"):
            _params_from_name("unknown-super-model-999b")

    def test_predict_from_name(self) -> None:
        model = MemoryModel()
        est = model.predict_from_name("gpt2", batch_size=2)
        assert est.peak_memory_mb > 0


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestProperties:
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        precision=st.sampled_from(["fp32", "fp16", "bf16"]),
    )
    @settings(max_examples=50)
    def test_memory_positive_and_monotone(
        self, batch_size: int, precision: str
    ) -> None:
        """Predicted memory is always positive and increases with batch_size."""
        model = MemoryModel()
        est_small = model.predict(125_000_000, max(1, batch_size - 1), precision, "adamw")
        est_large = model.predict(125_000_000, batch_size, precision, "adamw")

        assert est_small.peak_memory_mb > 0
        assert est_large.peak_memory_mb >= est_small.peak_memory_mb

    @given(
        param_count=st.integers(min_value=1_000_000, max_value=10_000_000_000),
    )
    @settings(max_examples=30)
    def test_larger_model_uses_more_memory(self, param_count: int) -> None:
        """More parameters always means more memory (all else equal)."""
        model = MemoryModel()
        small = model.predict(param_count, 1)
        large = model.predict(param_count * 2, 1)
        assert large.peak_memory_mb > small.peak_memory_mb


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------

class TestConfidenceInterval:
    def test_lower_less_than_upper(self) -> None:
        model = MemoryModel()
        est = model.predict(125_000_000, 4)
        assert est.lower_mb < est.peak_memory_mb < est.upper_mb

    def test_lower_non_negative(self) -> None:
        model = MemoryModel()
        est = model.predict(1_000, 1)
        assert est.lower_mb >= 0
