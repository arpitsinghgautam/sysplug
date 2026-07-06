"""Numeric/formula-level tests for MemoryModel.

These tests validate the exact math of each memory component against
hand-computed expected values.  No mocking — the real MemoryModel class
is exercised end-to-end with precise assertions.
"""

from __future__ import annotations

import math

import pytest

from sysplug.memory_model import MemoryBreakdown, MemoryModel, PrecisionMode, _params_from_name

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mb(param_count: int, bytes_per: float) -> float:
    """param_count × bytes_per / (1024²)."""
    return param_count * bytes_per / 1_048_576


# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

GPT2_PARAMS = 125_000_000  # 125 M
BYTES_BF16 = 2.0
BYTES_FP32 = 4.0
BYTES_INT4 = 0.5


# ---------------------------------------------------------------------------
# 1. Parameter memory
# ---------------------------------------------------------------------------


class TestParameterMemoryExact:
    def test_bf16_125m(self) -> None:
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="sgd", parallelism="none"
        )
        expected_params = _mb(GPT2_PARAMS, 2.0)
        assert math.isclose(est.breakdown.parameters_mb, expected_params, rel_tol=1e-6)

    def test_fp32_125m(self) -> None:
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="fp32", optimizer="sgd", parallelism="none"
        )
        expected_params = _mb(GPT2_PARAMS, 4.0)
        assert math.isclose(est.breakdown.parameters_mb, expected_params, rel_tol=1e-6)

    def test_int4_125m(self) -> None:
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="int4", optimizer="sgd", parallelism="none"
        )
        expected_params = _mb(GPT2_PARAMS, 0.5)
        assert math.isclose(est.breakdown.parameters_mb, expected_params, rel_tol=1e-6)

    def test_int8_125m(self) -> None:
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="int8", optimizer="sgd", parallelism="none"
        )
        expected_params = _mb(GPT2_PARAMS, 1.0)
        assert math.isclose(est.breakdown.parameters_mb, expected_params, rel_tol=1e-6)

    def test_fp16_equals_bf16(self) -> None:
        """FP16 and BF16 are both 2-byte formats — parameter memory must match."""
        model = MemoryModel(gpu_count=1)
        fp16 = model.predict(
            GPT2_PARAMS, batch_size=1, precision="fp16", optimizer="sgd", parallelism="none"
        )
        bf16 = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="sgd", parallelism="none"
        )
        assert fp16.breakdown.parameters_mb == bf16.breakdown.parameters_mb


# ---------------------------------------------------------------------------
# 2. Gradient memory
# ---------------------------------------------------------------------------


class TestGradientMemoryExact:
    def test_no_parallelism_grads_equal_params(self) -> None:
        """Without ZeRO/FSDP, gradients have same dtype × size as parameters."""
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="sgd", parallelism="none"
        )
        assert math.isclose(est.breakdown.gradients_mb, est.breakdown.parameters_mb, rel_tol=1e-6)

    def test_zero2_shards_gradients(self) -> None:
        """ZeRO-2 divides gradient memory by gpu_count."""
        gpu_count = 4
        model = MemoryModel(gpu_count=gpu_count)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="zero2"
        )
        full_grad = _mb(GPT2_PARAMS, 2.0)
        assert math.isclose(est.breakdown.gradients_mb, full_grad / gpu_count, rel_tol=1e-6)

    def test_zero3_shards_gradients(self) -> None:
        gpu_count = 8
        model = MemoryModel(gpu_count=gpu_count)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="zero3"
        )
        full_grad = _mb(GPT2_PARAMS, 2.0)
        assert math.isclose(est.breakdown.gradients_mb, full_grad / gpu_count, rel_tol=1e-6)

    def test_fsdp_shards_gradients(self) -> None:
        gpu_count = 4
        model = MemoryModel(gpu_count=gpu_count)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="fsdp"
        )
        full_grad = _mb(GPT2_PARAMS, 2.0)
        assert math.isclose(est.breakdown.gradients_mb, full_grad / gpu_count, rel_tol=1e-6)

    def test_zero1_does_not_shard_gradients(self) -> None:
        """ZeRO-1 only shards optimizer states, NOT gradients."""
        model = MemoryModel(gpu_count=4)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="zero1"
        )
        full_grad = _mb(GPT2_PARAMS, 2.0)
        assert math.isclose(est.breakdown.gradients_mb, full_grad, rel_tol=1e-6)

    def test_dp_does_not_shard_gradients(self) -> None:
        model = MemoryModel(gpu_count=4)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="dp"
        )
        full_grad = _mb(GPT2_PARAMS, 2.0)
        assert math.isclose(est.breakdown.gradients_mb, full_grad, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 3. Optimizer state memory
# ---------------------------------------------------------------------------


class TestOptimizerStatesExact:
    def _fp32_param_mb(self, param_count: int) -> float:
        return param_count * 4.0 / 1_048_576

    def test_adamw_three_fp32_copies(self) -> None:
        """AdamW = 2 momentum tensors (fp32) + 1 master weight copy (fp32)."""
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="none"
        )
        expected = 3.0 * self._fp32_param_mb(GPT2_PARAMS)
        assert math.isclose(est.breakdown.optimizer_states_mb, expected, rel_tol=1e-6)

    def test_adam_same_as_adamw(self) -> None:
        model = MemoryModel(gpu_count=1)
        adamw = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="none"
        )
        adam = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adam", parallelism="none"
        )
        assert math.isclose(
            adamw.breakdown.optimizer_states_mb, adam.breakdown.optimizer_states_mb, rel_tol=1e-6
        )

    def test_sgd_zero_states(self) -> None:
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="sgd", parallelism="none"
        )
        assert est.breakdown.optimizer_states_mb == 0.0

    def test_adafactor_half_fp32(self) -> None:
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adafactor", parallelism="none"
        )
        expected = 0.5 * self._fp32_param_mb(GPT2_PARAMS)
        assert math.isclose(est.breakdown.optimizer_states_mb, expected, rel_tol=1e-6)

    def test_zero1_shards_optimizer(self) -> None:
        gpu_count = 4
        model = MemoryModel(gpu_count=gpu_count)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="zero1"
        )
        full_opt = 3.0 * self._fp32_param_mb(GPT2_PARAMS)
        assert math.isclose(est.breakdown.optimizer_states_mb, full_opt / gpu_count, rel_tol=1e-6)

    def test_zero3_shards_optimizer(self) -> None:
        gpu_count = 8
        model = MemoryModel(gpu_count=gpu_count)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="zero3"
        )
        full_opt = 3.0 * self._fp32_param_mb(GPT2_PARAMS)
        assert math.isclose(est.breakdown.optimizer_states_mb, full_opt / gpu_count, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 4. Activation memory
# ---------------------------------------------------------------------------


class TestActivationMemoryExact:
    def _expected_act_mb(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        bytes_elem: float,
        use_gc: bool = False,
    ) -> float:
        act_per_layer = batch_size * seq_len * hidden_dim * 34 * bytes_elem
        total = act_per_layer * num_layers
        if use_gc:
            total *= math.sqrt(num_layers) / num_layers
        return total / 1_048_576

    def test_activation_scales_with_batch_size(self) -> None:
        """Doubling batch_size should double activation memory."""
        model = MemoryModel(gpu_count=1)
        est1 = model.predict(
            GPT2_PARAMS,
            batch_size=2,
            precision="bf16",
            optimizer="sgd",
            parallelism="none",
            hidden_dim=64,
            num_layers=2,
        )
        est2 = model.predict(
            GPT2_PARAMS,
            batch_size=4,
            precision="bf16",
            optimizer="sgd",
            parallelism="none",
            hidden_dim=64,
            num_layers=2,
        )
        ratio = est2.breakdown.activations_mb / est1.breakdown.activations_mb
        assert math.isclose(ratio, 2.0, rel_tol=1e-6)

    def test_activation_scales_with_seq_len(self) -> None:
        """Doubling sequence length should double activation memory."""
        model = MemoryModel(gpu_count=1)
        est1 = model.predict(
            GPT2_PARAMS,
            batch_size=2,
            precision="bf16",
            optimizer="sgd",
            parallelism="none",
            sequence_length=256,
            hidden_dim=64,
            num_layers=2,
        )
        est2 = model.predict(
            GPT2_PARAMS,
            batch_size=2,
            precision="bf16",
            optimizer="sgd",
            parallelism="none",
            sequence_length=512,
            hidden_dim=64,
            num_layers=2,
        )
        ratio = est2.breakdown.activations_mb / est1.breakdown.activations_mb
        assert math.isclose(ratio, 2.0, rel_tol=1e-6)

    def test_gradient_checkpointing_reduces_activations(self) -> None:
        """GC with 4 layers reduces activations by sqrt(4)/4 = 0.5."""
        model = MemoryModel(gpu_count=1)
        base = model.predict(
            GPT2_PARAMS,
            batch_size=4,
            precision="bf16",
            optimizer="sgd",
            parallelism="none",
            hidden_dim=64,
            num_layers=4,
        )
        gc = model.predict(
            GPT2_PARAMS,
            batch_size=4,
            precision="bf16",
            optimizer="sgd",
            parallelism="none",
            hidden_dim=64,
            num_layers=4,
            use_gradient_checkpointing=True,
        )
        expected_ratio = math.sqrt(4) / 4  # = 0.5
        actual_ratio = gc.breakdown.activations_mb / base.breakdown.activations_mb
        assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-6)

    def test_explicit_hidden_dim_and_layers(self) -> None:
        """When hidden_dim and num_layers are given explicitly, formula is exact."""
        batch_size = 4
        seq_len = 512
        hidden_dim = 256
        num_layers = 6
        bytes_elem = 2.0  # bf16

        model = MemoryModel(gpu_count=1)
        est = model.predict(
            param_count=GPT2_PARAMS,
            batch_size=batch_size,
            precision="bf16",
            optimizer="sgd",
            parallelism="none",
            sequence_length=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        expected = self._expected_act_mb(batch_size, seq_len, hidden_dim, num_layers, bytes_elem)
        assert math.isclose(est.breakdown.activations_mb, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 5. Framework overhead
# ---------------------------------------------------------------------------


class TestFrameworkOverhead:
    def test_overhead_always_500mb(self) -> None:
        model = MemoryModel(gpu_count=1)
        for prec in ("fp32", "bf16", "fp16", "int8", "int4"):
            est = model.predict(
                GPT2_PARAMS, batch_size=1, precision=prec, optimizer="sgd", parallelism="none"
            )
            assert est.breakdown.framework_overhead_mb == 500.0

    def test_overhead_independent_of_batch_size(self) -> None:
        model = MemoryModel(gpu_count=1)
        for bs in (1, 4, 16, 64):
            est = model.predict(
                GPT2_PARAMS, batch_size=bs, precision="bf16", optimizer="sgd", parallelism="none"
            )
            assert est.breakdown.framework_overhead_mb == 500.0


# ---------------------------------------------------------------------------
# 6. Total memory = sum of components
# ---------------------------------------------------------------------------


class TestTotalMemory:
    def test_total_equals_sum_of_breakdown(self) -> None:
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=8, precision="bf16", optimizer="adamw", parallelism="none"
        )
        bd = est.breakdown
        expected_total = (
            bd.parameters_mb
            + bd.gradients_mb
            + bd.optimizer_states_mb
            + bd.activations_mb
            + bd.framework_overhead_mb
        )
        assert math.isclose(bd.total_mb, expected_total, rel_tol=1e-9)

    def test_peak_memory_equals_calibrated_total(self) -> None:
        """peak_memory_mb = breakdown.total_mb × calibration_factor."""
        model = MemoryModel(gpu_count=1, calibration_factor=1.25)
        est = model.predict(
            GPT2_PARAMS, batch_size=4, precision="bf16", optimizer="adamw", parallelism="none"
        )
        assert math.isclose(est.peak_memory_mb, est.breakdown.total_mb * 1.25, rel_tol=1e-9)

    def test_confidence_interval_width(self) -> None:
        """CI half-width = 15% of peak_memory_mb."""
        model = MemoryModel(gpu_count=1)
        est = model.predict(
            GPT2_PARAMS, batch_size=4, precision="bf16", optimizer="adamw", parallelism="none"
        )
        half = est.peak_memory_mb * 0.15
        assert math.isclose(est.upper_mb - est.peak_memory_mb, half, rel_tol=1e-9)
        assert math.isclose(est.peak_memory_mb - est.lower_mb, half, rel_tol=1e-9)

    def test_lower_bound_non_negative(self) -> None:
        """Lower CI bound must never be negative."""
        model = MemoryModel(gpu_count=1)
        for params in (1_000, 1_000_000, 7_000_000_000):
            for bs in (1, 32):
                est = model.predict(
                    params, batch_size=bs, precision="bf16", optimizer="adamw", parallelism="none"
                )
                assert est.lower_mb >= 0.0


# ---------------------------------------------------------------------------
# 7. ZeRO-3 full-stack sharding
# ---------------------------------------------------------------------------


class TestZeROFullSharding:
    def test_zero3_parameters_sharded(self) -> None:
        gpu_count = 4
        model = MemoryModel(gpu_count=gpu_count)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="zero3"
        )
        expected_params = _mb(GPT2_PARAMS, 2.0) / gpu_count
        assert math.isclose(est.breakdown.parameters_mb, expected_params, rel_tol=1e-6)

    def test_fsdp_parameters_sharded(self) -> None:
        gpu_count = 4
        model = MemoryModel(gpu_count=gpu_count)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="fsdp"
        )
        expected_params = _mb(GPT2_PARAMS, 2.0) / gpu_count
        assert math.isclose(est.breakdown.parameters_mb, expected_params, rel_tol=1e-6)

    def test_zero2_parameters_not_sharded(self) -> None:
        """ZeRO-2 does NOT shard parameters, only gradients + optimizer states."""
        gpu_count = 4
        model = MemoryModel(gpu_count=gpu_count)
        est = model.predict(
            GPT2_PARAMS, batch_size=1, precision="bf16", optimizer="adamw", parallelism="zero2"
        )
        expected_params = _mb(GPT2_PARAMS, 2.0)  # full copy on every rank
        assert math.isclose(est.breakdown.parameters_mb, expected_params, rel_tol=1e-6)

    def test_zero3_total_lower_than_no_parallelism(self) -> None:
        """ZeRO-3 with 4 GPUs must fit in significantly less per-GPU memory."""
        model_single = MemoryModel(gpu_count=1)
        model_zero3 = MemoryModel(gpu_count=4)
        single = model_single.predict(GPT2_PARAMS, 4, "bf16", "adamw", "none")
        z3 = model_zero3.predict(GPT2_PARAMS, 4, "bf16", "adamw", "zero3")
        assert z3.peak_memory_mb < single.peak_memory_mb


# ---------------------------------------------------------------------------
# 8. MemoryBreakdown.to_dict()
# ---------------------------------------------------------------------------


class TestMemoryBreakdownDict:
    def test_all_keys_present(self) -> None:
        bd = MemoryBreakdown(
            parameters_mb=100.0,
            gradients_mb=100.0,
            optimizer_states_mb=300.0,
            activations_mb=50.0,
            framework_overhead_mb=500.0,
        )
        d = bd.to_dict()
        expected_keys = {
            "parameters_mb",
            "gradients_mb",
            "optimizer_states_mb",
            "activations_mb",
            "framework_overhead_mb",
            "total_mb",
        }
        assert set(d.keys()) == expected_keys

    def test_total_correct_in_dict(self) -> None:
        bd = MemoryBreakdown(
            parameters_mb=200.0,
            gradients_mb=200.0,
            optimizer_states_mb=600.0,
            activations_mb=40.0,
            framework_overhead_mb=500.0,
        )
        d = bd.to_dict()
        assert math.isclose(d["total_mb"], 1540.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 9. Calibration factor — exact least-squares
# ---------------------------------------------------------------------------


class TestCalibrationExact:
    def test_perfect_fit_gives_factor_one(self) -> None:
        """If measured == predicted, calibration factor must be 1.0."""
        model = MemoryModel(gpu_count=1)
        pred = model.predict(
            GPT2_PARAMS, batch_size=4, precision="bf16", optimizer="adamw", parallelism="none"
        )
        samples = [
            {
                "param_count": GPT2_PARAMS,
                "batch_size": 4,
                "precision": "bf16",
                "optimizer": "adamw",
                "parallelism": "none",
                "measured_mb": pred.peak_memory_mb,
            }
        ]
        factor = model.calibrate(samples)
        assert math.isclose(factor, 1.0, rel_tol=1e-6)

    def test_factor_scales_predictions(self) -> None:
        """After calibration with measured = 2×predicted, factor must be ≈2.0."""
        model = MemoryModel(gpu_count=1)
        pred = model.predict(
            GPT2_PARAMS, batch_size=4, precision="bf16", optimizer="adamw", parallelism="none"
        )
        samples = [
            {
                "param_count": GPT2_PARAMS,
                "batch_size": 4,
                "precision": "bf16",
                "optimizer": "adamw",
                "parallelism": "none",
                "measured_mb": pred.peak_memory_mb * 2.0,
            }
        ]
        factor = model.calibrate(samples)
        assert math.isclose(factor, 2.0, rel_tol=1e-4)

    def test_factor_clamped_at_0_1(self) -> None:
        """Calibration factor is clamped to min 0.1 (sanity guard)."""
        model = MemoryModel(gpu_count=1)
        samples = [
            {
                "param_count": GPT2_PARAMS,
                "batch_size": 1,
                "precision": "bf16",
                "optimizer": "adamw",
                "parallelism": "none",
                "measured_mb": 0.001,
            }
        ]
        factor = model.calibrate(samples)
        assert factor >= 0.1

    def test_calibration_affects_next_prediction(self) -> None:
        """A factor of 2 must double peak_memory_mb in the next prediction."""
        model = MemoryModel(gpu_count=1)
        pred_before = model.predict(
            GPT2_PARAMS, batch_size=4, precision="bf16", optimizer="adamw", parallelism="none"
        )
        samples = [
            {
                "param_count": GPT2_PARAMS,
                "batch_size": 4,
                "precision": "bf16",
                "optimizer": "adamw",
                "parallelism": "none",
                "measured_mb": pred_before.peak_memory_mb * 2.0,
            }
        ]
        model.calibrate(samples)
        pred_after = model.predict(
            GPT2_PARAMS, batch_size=4, precision="bf16", optimizer="adamw", parallelism="none"
        )
        assert math.isclose(
            pred_after.peak_memory_mb, pred_before.peak_memory_mb * 2.0, rel_tol=1e-4
        )

    def test_multi_sample_least_squares(self) -> None:
        """With multiple samples, the least-squares solution must be returned."""
        model = MemoryModel(gpu_count=1)
        samples = []
        raw_preds = []
        for bs in (1, 2, 4, 8):
            pred = model.predict(
                GPT2_PARAMS, batch_size=bs, precision="bf16", optimizer="adamw", parallelism="none"
            )
            raw_preds.append(pred.peak_memory_mb)
            # pretend measured = 1.5 × predicted
            samples.append(
                {
                    "param_count": GPT2_PARAMS,
                    "batch_size": bs,
                    "precision": "bf16",
                    "optimizer": "adamw",
                    "parallelism": "none",
                    "measured_mb": pred.peak_memory_mb * 1.5,
                }
            )
        factor = model.calibrate(samples)
        assert math.isclose(factor, 1.5, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 10. Model-name lookup
# ---------------------------------------------------------------------------


class TestModelNameLookup:
    @pytest.mark.parametrize(
        "name,expected_b",
        [
            ("gpt2", 0.117),
            ("llama-3-8b", 8.0),
            ("llama-2-70b", 70.0),
            ("bert-base", 0.110),
            ("mistral-7b", 7.0),
            ("phi-2", 2.7),
        ],
    )
    def test_known_models(self, name, expected_b) -> None:
        count = _params_from_name(name)
        assert count == int(expected_b * 1e9)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model name"):
            _params_from_name("my-custom-model-99b")

    def test_predict_from_name_gpt2(self) -> None:
        model = MemoryModel(gpu_count=1)
        est = model.predict_from_name("gpt2", batch_size=4)
        assert est.peak_memory_mb > 0

    def test_predict_from_name_case_insensitive(self) -> None:
        model = MemoryModel(gpu_count=1)
        # upper-case name should resolve via lower().strip()
        est = model.predict_from_name("GPT2", batch_size=4)
        assert est.peak_memory_mb > 0


# ---------------------------------------------------------------------------
# 11. PrecisionMode enum
# ---------------------------------------------------------------------------


class TestPrecisionMode:
    def test_enum_values(self) -> None:
        assert PrecisionMode.FP32 == "fp32"
        assert PrecisionMode.FP16 == "fp16"
        assert PrecisionMode.BF16 == "bf16"
        assert PrecisionMode.INT8 == "int8"
        assert PrecisionMode.INT4 == "int4"

    def test_predict_accepts_string_precision(self) -> None:
        model = MemoryModel()
        for prec in ("fp32", "fp16", "bf16", "int8", "int4"):
            est = model.predict(
                GPT2_PARAMS, batch_size=1, precision=prec, optimizer="sgd", parallelism="none"
            )
            assert est.peak_memory_mb > 0

    def test_predict_invalid_precision_raises(self) -> None:
        model = MemoryModel()
        with pytest.raises(ValueError):
            model.predict(
                GPT2_PARAMS, batch_size=1, precision="fp64", optimizer="sgd", parallelism="none"
            )


# ---------------------------------------------------------------------------
# 12. MemoryModel properties
# ---------------------------------------------------------------------------


class TestMemoryModelProperties:
    def test_gpu_count_property(self) -> None:
        model = MemoryModel(gpu_count=4)
        assert model.gpu_count == 4

    def test_calibration_factor_default(self) -> None:
        model = MemoryModel()
        assert model.calibration_factor == 1.0

    def test_calibration_factor_constructor(self) -> None:
        model = MemoryModel(calibration_factor=1.5)
        assert model.calibration_factor == 1.5

    def test_invalid_gpu_count_raises(self) -> None:
        with pytest.raises(ValueError, match="gpu_count"):
            MemoryModel(gpu_count=0)

    def test_calibrate_empty_raises(self) -> None:
        model = MemoryModel()
        with pytest.raises(ValueError, match="empty"):
            model.calibrate([])
