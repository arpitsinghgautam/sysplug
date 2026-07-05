"""Unit tests for ThroughputModel."""

from __future__ import annotations

import pytest

from sysplug.throughput_model import ThroughputModel, _flops_per_step


class TestFlopsPerStep:
    def test_standard_formula(self) -> None:
        """FLOPs = 6 × params × seq_len × batch."""
        flops = _flops_per_step(7_000_000_000, batch_size=4, sequence_length=512)
        expected = 6 * 7_000_000_000 * 512 * 4
        assert flops == pytest.approx(expected, rel=1e-6)

    def test_single_sample(self) -> None:
        flops = _flops_per_step(1_000_000, batch_size=1, sequence_length=1)
        assert flops == pytest.approx(6 * 1_000_000, rel=1e-6)


class TestThroughputModelBasic:
    def test_positive_throughput(self) -> None:
        model = ThroughputModel(gpu_name="A100")
        est = model.predict(
            effective_batch_size=32,
            model_size_params=125_000_000,
            precision="bf16",
        )
        assert est.samples_per_sec > 0
        assert est.tokens_per_sec > 0

    def test_tokens_per_sec_consistent(self) -> None:
        """tokens/sec = samples/sec × seq_len."""
        model = ThroughputModel(gpu_name="A100")
        seq_len = 256
        est = model.predict(32, 125_000_000, "bf16", sequence_length=seq_len)
        assert est.tokens_per_sec == pytest.approx(est.samples_per_sec * seq_len, rel=1e-6)

    def test_larger_batch_nondecreasing_throughput(self) -> None:
        """Larger effective batch should not decrease throughput (compute-bound: equal)."""
        model = ThroughputModel(gpu_name="A100")
        small = model.predict(4, 125_000_000, "bf16")
        large = model.predict(32, 125_000_000, "bf16")
        # In compute-bound regime, samples/sec is equal; in memory-bound it increases.
        assert large.samples_per_sec >= small.samples_per_sec

    def test_invalid_batch_size_raises(self) -> None:
        model = ThroughputModel()
        with pytest.raises(ValueError, match="effective_batch_size"):
            model.predict(0, 125_000_000)

    def test_invalid_model_size_raises(self) -> None:
        model = ThroughputModel()
        with pytest.raises(ValueError, match="model_size_params"):
            model.predict(8, 0)

    def test_gpu_count_scales_throughput(self) -> None:
        """More GPUs should increase throughput approximately linearly."""
        single = ThroughputModel(gpu_name="A100", gpu_count=1)
        multi = ThroughputModel(gpu_name="A100", gpu_count=4)
        est1 = single.predict(32, 125_000_000, "bf16")
        est4 = multi.predict(32, 125_000_000, "bf16")
        assert est4.samples_per_sec > est1.samples_per_sec


class TestRooflineMetrics:
    def test_arithmetic_intensity_positive(self) -> None:
        model = ThroughputModel(gpu_name="A100")
        est = model.predict(32, 7_000_000_000, "bf16")
        assert est.arithmetic_intensity > 0

    def test_attainable_tflops_positive(self) -> None:
        model = ThroughputModel(gpu_name="A100")
        est = model.predict(32, 1_000_000_000, "bf16")
        assert est.attainable_tflops > 0

    def test_is_memory_bound_field_is_bool(self) -> None:
        model = ThroughputModel()
        est = model.predict(8, 125_000_000, "bf16")
        assert isinstance(est.is_memory_bound, bool)


class TestCalibration:
    def test_calibrate_roofline(self) -> None:
        model = ThroughputModel(gpu_name="A100")
        samples = [
            {
                "effective_batch_size": 32,
                "model_size_params": 125_000_000,
                "precision": "bf16",
                "measured_samples_per_sec": 50.0,
            }
        ]
        factor = model.calibrate_roofline(samples)
        assert 0.01 < factor < 10.0

    def test_calibrate_empty_raises(self) -> None:
        model = ThroughputModel()
        with pytest.raises(ValueError, match="must not be empty"):
            model.calibrate_roofline([])

    def test_fit_empirical_requires_two_samples(self) -> None:
        model = ThroughputModel()
        with pytest.raises(ValueError, match="at least 2"):
            model.fit_empirical([{"effective_batch_size": 8, "measured_samples_per_sec": 10}])

    def test_fit_empirical_uses_linear_model(self) -> None:
        model = ThroughputModel(gpu_name="A100")
        samples = [
            {"effective_batch_size": 8, "measured_samples_per_sec": 20.0},
            {"effective_batch_size": 16, "measured_samples_per_sec": 40.0},
            {"effective_batch_size": 32, "measured_samples_per_sec": 80.0},
        ]
        model.fit_empirical(samples)
        # After fitting, predictions should be close to measured
        est = model.predict(16, 125_000_000, "bf16")
        # Should be in the ballpark of 40 samples/sec
        assert 10 < est.samples_per_sec < 200


class TestGPULookup:
    @pytest.mark.parametrize("gpu_name,prec", [
        ("A100", "bf16"),
        ("V100", "fp16"),
        ("T4", "fp16"),
        ("RTX 4090", "bf16"),
    ])
    def test_known_gpu_positive_throughput(self, gpu_name: str, prec: str) -> None:
        model = ThroughputModel(gpu_name=gpu_name)
        est = model.predict(16, 125_000_000, prec)
        assert est.samples_per_sec > 0

    def test_unknown_gpu_uses_default(self) -> None:
        model = ThroughputModel(gpu_name="unknown-gpu-xyzzy")
        est = model.predict(8, 125_000_000, "fp32")
        assert est.samples_per_sec > 0
