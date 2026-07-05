"""Real computation tests — no mocking of models, no GPU required.

These tests run the full SysPlug stack with real arithmetic, verifying
end-to-end behaviour that integration tests with hardware mocks can't
catch: actual numeric outputs, internal consistency, idempotency, etc.

A physical CUDA GPU is NOT required.  All tests run in CPU mode.
"""

from __future__ import annotations

import math
import time

import pytest

import sysplug
from sysplug.hardware import GPUSnapshot, HardwareSnapshot
from sysplug.memory_model import MemoryModel
from sysplug.solver import ConfigSolver, SolverConstraints
from sysplug.stability import StabilitySignal
from sysplug.throughput_model import ThroughputModel
from sysplug.utils.scaling_rules import (
    linear_lr_scale,
    recommended_lr_rule,
    sqrt_lr_scale,
    warmup_steps_for_batch,
)


# ---------------------------------------------------------------------------
# Hardware fixtures
# ---------------------------------------------------------------------------

def _hw(total_mb: float = 40_960, name: str = "A100") -> HardwareSnapshot:
    return HardwareSnapshot(
        gpus=[GPUSnapshot(0, name, total_mb, 0, total_mb, 0, 0, (8, 0), 2039)],
        cpu_count=8, ram_total_mb=65_536,
    )


# ---------------------------------------------------------------------------
# 1. Scaling rules — exact math
# ---------------------------------------------------------------------------

class TestScalingRulesMath:

    def test_linear_scale_doubles_lr_for_doubled_batch(self):
        new_lr = linear_lr_scale(base_lr=1e-4, base_batch=64, new_batch=128)
        assert math.isclose(new_lr, 2e-4, rel_tol=1e-9)

    def test_linear_scale_halves_lr_for_halved_batch(self):
        new_lr = linear_lr_scale(base_lr=1e-4, base_batch=128, new_batch=64)
        assert math.isclose(new_lr, 5e-5, rel_tol=1e-9)

    def test_sqrt_scale_sqrt2_increase_for_doubled_batch(self):
        new_lr = sqrt_lr_scale(base_lr=1e-4, base_batch=64, new_batch=128)
        expected = 1e-4 * math.sqrt(128 / 64)
        assert math.isclose(new_lr, expected, rel_tol=1e-9)

    def test_sqrt_always_less_than_linear_for_large_batch_increase(self):
        """Sqrt rule is more conservative than linear for any batch increase."""
        for factor in (2, 4, 8, 16):
            linear = linear_lr_scale(1e-4, 32, 32 * factor)
            sq = sqrt_lr_scale(1e-4, 32, 32 * factor)
            assert sq < linear

    def test_warmup_steps_inversely_proportional_to_batch(self):
        """Doubling batch size should halve warmup steps."""
        w1 = warmup_steps_for_batch(100, base_batch=32, new_batch=32)
        w2 = warmup_steps_for_batch(100, base_batch=32, new_batch=64)
        assert w2 == w1 // 2

    def test_recommended_rule_sft_small_batch(self):
        assert recommended_lr_rule("sft", batch_size=128) == "linear"

    def test_recommended_rule_sft_large_batch(self):
        assert recommended_lr_rule("sft", batch_size=512) == "sqrt"

    def test_recommended_rule_rlhf_always_sqrt(self):
        for batch in (8, 32, 128, 512):
            assert recommended_lr_rule("rlhf", batch_size=batch) == "sqrt"

    def test_recommended_rule_grpo_always_sqrt(self):
        assert recommended_lr_rule("grpo", batch_size=64) == "sqrt"

    def test_linear_scale_same_batch_returns_same_lr(self):
        lr = linear_lr_scale(2e-5, 64, 64)
        assert math.isclose(lr, 2e-5, rel_tol=1e-9)

    def test_sqrt_scale_same_batch_returns_same_lr(self):
        lr = sqrt_lr_scale(2e-5, 64, 64)
        assert math.isclose(lr, 2e-5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 2. Memory model — real predictions, monotonicity
# ---------------------------------------------------------------------------

class TestMemoryModelReal:

    def test_more_params_more_memory(self):
        """Monotonicity: larger model → more memory, always."""
        mm = MemoryModel(gpu_count=1)
        sizes = [10_000_000, 100_000_000, 1_000_000_000, 7_000_000_000]
        preds = [mm.predict(p, batch_size=4, precision="bf16",
                            optimizer="adamw").peak_memory_mb
                 for p in sizes]
        assert preds == sorted(preds), f"Not monotonically increasing: {preds}"

    def test_more_batch_more_memory(self):
        mm = MemoryModel(gpu_count=1)
        batches = [1, 2, 4, 8, 16, 32]
        preds = [mm.predict(125_000_000, bs, "bf16", "adamw").peak_memory_mb
                 for bs in batches]
        assert preds == sorted(preds), f"Not monotonically increasing: {preds}"

    def test_adamw_more_memory_than_sgd(self):
        mm = MemoryModel(gpu_count=1)
        sgd   = mm.predict(125_000_000, 4, "bf16", "sgd").peak_memory_mb
        adamw = mm.predict(125_000_000, 4, "bf16", "adamw").peak_memory_mb
        assert adamw > sgd

    def test_fp32_more_memory_than_int4(self):
        mm = MemoryModel(gpu_count=1)
        fp32 = mm.predict(125_000_000, 4, "fp32", "adamw").peak_memory_mb
        int4 = mm.predict(125_000_000, 4, "int4", "adamw").peak_memory_mb
        assert fp32 > int4

    def test_no_parallelism_more_memory_than_zero3(self):
        mm_single = MemoryModel(gpu_count=1)
        mm_zero3  = MemoryModel(gpu_count=8)
        single = mm_single.predict(7_000_000_000, 4, "bf16", "adamw",
                                   "none").peak_memory_mb
        z3     = mm_zero3.predict(7_000_000_000, 4, "bf16", "adamw",
                                   "zero3").peak_memory_mb
        assert single > z3

    def test_gc_reduces_memory(self):
        mm = MemoryModel(gpu_count=1)
        no_gc = mm.predict(125_000_000, 32, "bf16", "adamw",
                           use_gradient_checkpointing=False).peak_memory_mb
        with_gc = mm.predict(125_000_000, 32, "bf16", "adamw",
                              use_gradient_checkpointing=True).peak_memory_mb
        assert with_gc < no_gc

    def test_calibration_roundtrip(self):
        """Calibrate to 1.3× factor, then prediction should be 1.3× original."""
        mm = MemoryModel(gpu_count=1)
        pred_base = mm.predict(125_000_000, 4, "bf16", "adamw").peak_memory_mb
        samples = [{"param_count": 125_000_000, "batch_size": 4,
                    "precision": "bf16", "optimizer": "adamw",
                    "parallelism": "none",
                    "measured_mb": pred_base * 1.3}]
        factor = mm.calibrate(samples)
        pred_after = mm.predict(125_000_000, 4, "bf16", "adamw").peak_memory_mb
        assert math.isclose(pred_after, pred_base * 1.3, rel_tol=1e-3)
        assert math.isclose(factor, 1.3, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# 3. Throughput model — real predictions
# ---------------------------------------------------------------------------

class TestThroughputModelReal:

    def test_predicts_positive_throughput(self):
        tm = ThroughputModel(gpu_name="A100", gpu_count=1)
        est = tm.predict(32, 125_000_000, "bf16", 512)
        assert est.samples_per_sec > 0

    def test_tokens_per_sec_consistent(self):
        tm = ThroughputModel(gpu_name="A100", gpu_count=1)
        est = tm.predict(32, 125_000_000, "bf16", 512)
        expected = est.samples_per_sec * 512
        assert math.isclose(est.tokens_per_sec, expected, rel_tol=1e-9)

    def test_more_gpus_more_throughput(self):
        for n in (1, 2, 4, 8):
            tm = ThroughputModel(gpu_name="A100", gpu_count=n)
            est = tm.predict(32, 125_000_000, "bf16", 512)
            assert est.samples_per_sec > 0

        t1 = ThroughputModel(gpu_name="A100", gpu_count=1)
        t4 = ThroughputModel(gpu_name="A100", gpu_count=4)
        e1 = t1.predict(32, 125_000_000, "bf16", 512)
        e4 = t4.predict(32, 125_000_000, "bf16", 512)
        assert e4.samples_per_sec > e1.samples_per_sec

    def test_roofline_fields_on_estimate(self):
        """Roofline metrics are embedded in the ThroughputEstimate."""
        tm = ThroughputModel(gpu_name="A100", gpu_count=1)
        est = tm.predict(32, 125_000_000, "bf16", 512)
        assert hasattr(est, "arithmetic_intensity")
        assert hasattr(est, "attainable_tflops")
        assert hasattr(est, "is_memory_bound")
        assert est.attainable_tflops > 0
        assert est.arithmetic_intensity > 0

    def test_known_gpu_lookup(self):
        for gpu_name in ("A100", "V100", "T4", "RTX 4090"):
            tm = ThroughputModel(gpu_name=gpu_name, gpu_count=1)
            est = tm.predict(8, 125_000_000, "bf16", 512)
            assert est.samples_per_sec > 0

    def test_unknown_gpu_name_uses_default(self):
        tm = ThroughputModel(gpu_name="MyCustomGPU-9000", gpu_count=1)
        est = tm.predict(8, 125_000_000, "bf16", 512)
        assert est.samples_per_sec > 0


# ---------------------------------------------------------------------------
# 4. Stability signal — real convergence scenarios
# ---------------------------------------------------------------------------

class TestStabilityRealScenarios:

    def test_bert_style_training_stable(self):
        """Simulate a typical decreasing loss curve — should be stable."""
        sig = StabilitySignal(window_size=50)
        # Typical BERT loss: starts high, decreases, plateaus
        for step in range(50):
            loss = 4.0 * math.exp(-step * 0.08) + 0.5
            sig.record_loss(step, loss)
        report = sig.check()
        assert not report.is_diverging
        assert report.recommended_action in {"ok", "reduce_batch"}

    def test_lr_warmup_not_flagged_as_diverging(self):
        """Loss increases briefly during warmup then drops — must not be flagged."""
        sig = StabilitySignal(window_size=20, diverge_threshold=0.30)
        # Warmup: loss goes up slightly (< 30% increase), then drops
        losses = [2.0 + 0.01 * i for i in range(5)]  # +5% increase
        losses += [2.0 - 0.05 * i for i in range(15)]  # then drops
        for step, loss in enumerate(losses):
            sig.record_loss(step, loss)
        report = sig.check()
        # After the window sees mostly decreasing data, should not diverge
        assert not report.is_diverging

    def test_exploding_loss_detected(self):
        """Explosive gradient causing NaN-free divergence must be caught."""
        sig = StabilitySignal(window_size=20)
        for step in range(10):
            sig.record_loss(step, 1.0)  # stable
        for step in range(10, 20):
            sig.record_loss(step, 1.0 * (2 ** (step - 9)))  # doubles each step
        report = sig.check()
        assert report.is_diverging

    def test_oscillating_lr_schedule_detected(self):
        """Cyclic LR causing big oscillations should be flagged."""
        sig = StabilitySignal(window_size=20, oscillate_threshold=0.05)
        for step in range(20):
            # Oscillates between 0.5 and 2.5: mean=1.5, var=1.0, norm_var≈0.44
            loss = 0.5 if step % 2 == 0 else 2.5
            sig.record_loss(step, loss)
        report = sig.check()
        assert report.is_oscillating

    def test_multiple_check_calls_are_idempotent(self):
        """Calling check() multiple times without new data returns same result."""
        sig = StabilitySignal(window_size=10)
        for step in range(10):
            sig.record_loss(step, 1.0)
        r1 = sig.check()
        r2 = sig.check()
        assert r1.is_diverging == r2.is_diverging
        assert r1.is_oscillating == r2.is_oscillating
        assert r1.recommended_action == r2.recommended_action

    def test_window_rollover_correct(self):
        """Data older than window_size must not affect current check."""
        sig = StabilitySignal(window_size=10)
        # Fill with diverging data (high values)
        for step in range(10):
            sig.record_loss(step, 10.0 + step)
        # Overwrite with stable data
        for step in range(10, 20):
            sig.record_loss(step, 1.0)
        report = sig.check()
        # Window now contains only 1.0s → stable
        assert not report.is_diverging


# ---------------------------------------------------------------------------
# 5. Advisor — full end-to-end real runs
# ---------------------------------------------------------------------------

class TestAdvisorEndToEnd:

    def test_gpt2_on_a100_fits_in_memory(self):
        hw = _hw(40_960)
        adv = sysplug.Advisor(model="gpt2", hardware=hw, verbose=False)
        cfg = adv.suggest_config({"batch_size": 8, "optimizer": "adamw",
                                   "precision": "bf16"})
        budget = 40_960 * 0.85
        assert cfg.predicted_peak_memory_mb <= budget, (
            f"Config does not fit: {cfg.predicted_peak_memory_mb:.0f} > {budget:.0f} MB"
        )

    def test_repeated_suggest_is_deterministic(self):
        """Same inputs → same outputs every time."""
        hw = _hw(40_960)
        adv = sysplug.Advisor(model="gpt2", hardware=hw, verbose=False)
        cfg1 = adv.suggest_config({"batch_size": 4, "precision": "bf16",
                                    "optimizer": "adamw", "learning_rate": 2e-5})
        cfg2 = adv.suggest_config({"batch_size": 4, "precision": "bf16",
                                    "optimizer": "adamw", "learning_rate": 2e-5})
        assert cfg1.batch_size == cfg2.batch_size
        assert cfg1.precision == cfg2.precision
        assert math.isclose(cfg1.learning_rate, cfg2.learning_rate, rel_tol=1e-9)
        assert math.isclose(cfg1.predicted_peak_memory_mb,
                            cfg2.predicted_peak_memory_mb, rel_tol=1e-9)

    def test_what_if_consistent_with_suggest(self):
        """what_if with no change should produce same config as suggest_config."""
        hw = _hw(40_960)
        adv = sysplug.Advisor(model="gpt2", hardware=hw, verbose=False)
        cfg = adv.suggest_config({"batch_size": 4, "precision": "bf16",
                                   "optimizer": "adamw"})
        result = adv.what_if({})  # empty change
        # No changes → new config should equal original
        new = result.new_config
        assert new.batch_size == cfg.batch_size
        assert new.precision == cfg.precision

    def test_to_dict_round_trips_batch_size(self):
        hw = _hw(40_960)
        adv = sysplug.Advisor(model="gpt2", hardware=hw, verbose=False)
        cfg = adv.suggest_config({"batch_size": 4})
        d = cfg.to_dict()
        assert d["batch_size"] == cfg.batch_size

    def test_monitor_runs_without_error(self):
        hw = _hw(40_960)
        adv = sysplug.Advisor(model="gpt2", hardware=hw, verbose=False)
        adv.suggest_config({"batch_size": 4})
        with adv.monitor(check_interval_steps=5) as mon:
            for step in range(10):
                mon.record(step=step, loss=1.0 - step * 0.05)
            time.sleep(0.05)  # let background thread run at least once
        events = mon.get_events()
        assert isinstance(events, list)

    def test_config_to_deepspeed_bf16(self):
        hw = _hw(40_960)
        adv = sysplug.Advisor(model="gpt2", hardware=hw, verbose=False)
        cfg = adv.suggest_config({"precision": "bf16"})
        ds = cfg.to_deepspeed_config()
        assert isinstance(ds, dict)
        assert "train_micro_batch_size_per_gpu" in ds

    def test_config_apply_to_optimizer(self):
        try:
            import torch
            hw = _hw(40_960)
            adv = sysplug.Advisor(model="gpt2", hardware=hw, verbose=False)
            cfg = adv.suggest_config({"learning_rate": 3e-5})
            model = torch.nn.Linear(4, 2)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            cfg.apply_to_optimizer(opt)
            assert math.isclose(opt.param_groups[0]["lr"],
                                cfg.learning_rate, rel_tol=1e-9)
        except ImportError:
            pytest.skip("torch not available")


# ---------------------------------------------------------------------------
# 6. Config serialisation consistency
# ---------------------------------------------------------------------------

class TestConfigSerialisation:

    def test_to_dict_all_values_match_attributes(self):
        hw = _hw(40_960)
        adv = sysplug.Advisor(model="gpt2", hardware=hw, verbose=False)
        cfg = adv.suggest_config({"batch_size": 8, "precision": "bf16",
                                   "optimizer": "adamw"})
        d = cfg.to_dict()
        assert d["batch_size"]          == cfg.batch_size
        assert d["gradient_accumulation"] == cfg.gradient_accumulation
        assert d["effective_batch_size"] == cfg.effective_batch_size
        assert d["learning_rate"]       == cfg.learning_rate
        assert d["precision"]           == cfg.precision
        assert d["optimizer"]           == cfg.optimizer
        assert d["parallelism"]         == cfg.parallelism
        assert d["use_gradient_checkpointing"] == cfg.use_gradient_checkpointing
        assert d["warnings"]            == list(cfg.warnings)
        assert d["notes"]               == list(cfg.notes)

    def test_deepspeed_zero2_config(self):
        from sysplug.config import SysPlugConfig
        cfg = SysPlugConfig(
            batch_size=4, gradient_accumulation=2, precision="bf16",
            parallelism="zero2", gpu_count=4,
        )
        ds = cfg.to_deepspeed_config()
        assert ds["train_micro_batch_size_per_gpu"] == 4
        assert ds["gradient_accumulation_steps"] == 2
        assert ds["train_batch_size"] == 4 * 2 * 4
        assert ds.get("bf16", {}).get("enabled") is True
        assert ds.get("zero_optimization", {}).get("stage") == 2

    def test_deepspeed_zero3_config(self):
        from sysplug.config import SysPlugConfig
        cfg = SysPlugConfig(batch_size=2, parallelism="zero3", gpu_count=8,
                            precision="fp16")
        ds = cfg.to_deepspeed_config()
        assert ds.get("zero_optimization", {}).get("stage") == 3
        assert ds.get("fp16", {}).get("enabled") is True

    def test_deepspeed_no_zero_config(self):
        from sysplug.config import SysPlugConfig
        cfg = SysPlugConfig(batch_size=4, parallelism="none", precision="fp32")
        ds = cfg.to_deepspeed_config()
        assert "zero_optimization" not in ds
        assert "bf16" not in ds
        assert "fp16" not in ds

    def test_deepspeed_base_config_merged(self):
        from sysplug.config import SysPlugConfig
        cfg = SysPlugConfig(batch_size=4, precision="bf16", parallelism="none")
        base = {"scheduler": {"type": "WarmupLR"}}
        ds = cfg.to_deepspeed_config(base_config=base)
        assert "scheduler" in ds
        assert ds["train_micro_batch_size_per_gpu"] == 4
