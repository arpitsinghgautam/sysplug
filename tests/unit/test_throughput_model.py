"""Unit tests for ThroughputModel."""

from __future__ import annotations

import pytest

from sysplug.memory_model import PrecisionMode
from sysplug.throughput_model import (
    ThroughputModel,
    _flops_per_step,
    _get_gpu_spec,
    _GPUSpec,
    _infer_hidden_layers,
    _peak_tflops,
)


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
    @pytest.mark.parametrize(
        "gpu_name,prec",
        [
            ("A100", "bf16"),
            ("V100", "fp16"),
            ("T4", "fp16"),
            ("RTX 4090", "bf16"),
        ],
    )
    def test_known_gpu_positive_throughput(self, gpu_name: str, prec: str) -> None:
        model = ThroughputModel(gpu_name=gpu_name)
        est = model.predict(16, 125_000_000, prec)
        assert est.samples_per_sec > 0

    def test_unknown_gpu_uses_default(self) -> None:
        model = ThroughputModel(gpu_name="unknown-gpu-xyzzy")
        est = model.predict(8, 125_000_000, "fp32")
        assert est.samples_per_sec > 0


class TestBatchDependence:
    """Regression guards for the batch-invariance bug (throughput must depend
    on batch size: ramp at small batch, plateau at large batch)."""

    def test_throughput_is_not_batch_invariant(self) -> None:
        """The core regression: samples/sec at batch 1 and 32 must differ.

        The old roofline had ``bytes_traffic = 2·P·B`` which cancelled batch
        out of arithmetic intensity AND out of samples/sec, so every batch
        size returned the identical number. This must never come back.
        """
        model = ThroughputModel(gpu_name="A100")
        one = model.predict(1, 125_000_000, "bf16").samples_per_sec
        many = model.predict(32, 125_000_000, "bf16").samples_per_sec
        assert many > one * 1.2  # small batch is meaningfully slower

    def test_throughput_monotonic_nondecreasing_in_batch(self) -> None:
        model = ThroughputModel(gpu_name="A100")
        sps = [
            model.predict(b, 125_000_000, "bf16").samples_per_sec for b in (1, 2, 4, 8, 16, 32, 64)
        ]
        for lo, hi in zip(sps, sps[1:]):
            assert hi >= lo - 1e-6

    def test_throughput_saturates_at_large_batch(self) -> None:
        """Marginal gain must shrink as batch grows (plateau, not linear)."""
        model = ThroughputModel(gpu_name="A100")
        s1 = model.predict(1, 125_000_000, "bf16").samples_per_sec
        s2 = model.predict(2, 125_000_000, "bf16").samples_per_sec
        s128 = model.predict(128, 125_000_000, "bf16").samples_per_sec
        s256 = model.predict(256, 125_000_000, "bf16").samples_per_sec
        early_gain = s2 - s1
        late_gain = s256 - s128
        assert late_gain < early_gain

    def test_arithmetic_intensity_grows_with_batch(self) -> None:
        """Weights are reused across the batch, so AI must increase with batch."""
        model = ThroughputModel(gpu_name="A100")
        ai_small = model.predict(1, 1_000_000_000, "bf16").arithmetic_intensity
        ai_large = model.predict(64, 1_000_000_000, "bf16").arithmetic_intensity
        assert ai_large > ai_small

    def test_step_time_increases_with_batch(self) -> None:
        model = ThroughputModel(gpu_name="A100")
        t_small = model.predict(1, 125_000_000, "bf16").step_time_sec
        t_large = model.predict(32, 125_000_000, "bf16").step_time_sec
        assert t_large > t_small > 0


class TestGPUSpecTable:
    """Guards for the GPU spec-table data bugs."""

    @pytest.mark.parametrize("gpu_name", ["RTX 3090", "RTX 3080"])
    def test_ampere_consumer_bf16_not_zero(self, gpu_name: str) -> None:
        """3090/3080 bf16 was 0.0 → zeroed throughput. Must be positive now."""
        model = ThroughputModel(gpu_name=gpu_name)
        est = model.predict(16, 125_000_000, "bf16")
        assert est.samples_per_sec > 0

    def test_blackwell_rtx_pro_5000_resolves_to_real_spec(self) -> None:
        """The user's card must not fall through to the default spec."""
        spec = _get_gpu_spec("NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU")
        assert spec.name_key != "unknown"
        assert spec.tflops_bf16 > 20.0

    def test_peak_tflops_falls_back_when_bf16_missing(self) -> None:
        """A stale spec with bf16=0 must fall back, never return 0."""
        stale = _GPUSpec("stale", tflops_fp32=10.0, tflops_fp16=50.0, tflops_bf16=0.0)
        assert _peak_tflops(stale, PrecisionMode.BF16) == 50.0


class TestArchInference:
    """`_infer_hidden_layers` must produce realistic transformer dimensions."""

    @pytest.mark.parametrize(
        "params,h_lo,h_hi,l_lo,l_hi",
        [
            (125_000_000, 640, 1280, 6, 14),  # gpt2-small ~ 768/12
            (7_000_000_000, 3200, 4800, 24, 48),  # llama-7b ~ 4096/32
            (70_000_000_000, 7000, 10000, 60, 96),  # llama-70b ~ 8192/80
        ],
    )
    def test_inference_in_realistic_range(
        self, params: int, h_lo: int, h_hi: int, l_lo: int, l_hi: int
    ) -> None:
        hidden, layers = _infer_hidden_layers(params)
        assert h_lo <= hidden <= h_hi
        assert l_lo <= layers <= l_hi

    def test_recovers_param_count_within_2x(self) -> None:
        """params ≈ 12·L·H² should round-trip within a factor of 2."""
        for params in (125_000_000, 1_300_000_000, 7_000_000_000):
            hidden, layers = _infer_hidden_layers(params)
            recovered = 12 * layers * hidden * hidden
            assert 0.5 < recovered / params < 2.0


class TestEmpiricalStepTimeModel:
    def test_empirical_reproduces_saturating_curve(self) -> None:
        """Fitting a saturating step-time series must plateau, not extrapolate
        linearly to infinity."""
        model = ThroughputModel(gpu_name="A100")
        # step_time ≈ 0.01·batch + 0.05  →  saturating throughput
        samples = [
            {"effective_batch_size": b, "measured_samples_per_sec": b / (0.01 * b + 0.05)}
            for b in (1, 2, 4, 8, 16, 32)
        ]
        model.fit_empirical(samples)
        s32 = model.predict(32, 125_000_000, "bf16").samples_per_sec
        s256 = model.predict(256, 125_000_000, "bf16").samples_per_sec
        # Plateau: doubling batch 8× must NOT multiply throughput 8×.
        assert s256 < 4 * s32
        # And it should approach the asymptote 1/0.01 = 100 samp/s.
        assert 80 < s256 < 100

    def test_fit_empirical_rejects_nonpositive(self) -> None:
        model = ThroughputModel()
        with pytest.raises(ValueError):
            model.fit_empirical(
                [
                    {"effective_batch_size": 8, "measured_samples_per_sec": 0.0},
                    {"effective_batch_size": 16, "measured_samples_per_sec": 40.0},
                ]
            )
