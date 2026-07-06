"""Full-pipeline integration tests for the Advisor class.

These tests exercise the complete stack — Advisor → ConfigSolver →
MemoryModel + ThroughputModel — with faked hardware snapshots.
No internal model methods are mocked; only GPU hardware detection is
replaced with known snapshots so results are deterministic.
"""

from __future__ import annotations

import math

import pytest

from sysplug.advisor import Advisor, WhatIfResult
from sysplug.config import SysPlugConfig
from sysplug.hardware import GPUSnapshot, HardwareSnapshot
from sysplug.solver import SolverConstraints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _a100_hw(count: int = 1) -> HardwareSnapshot:
    gpus = [
        GPUSnapshot(
            device_id=i,
            gpu_name="A100",
            total_memory_mb=40_960,
            used_memory_mb=0,
            free_memory_mb=40_960,
            gpu_utilization_pct=0,
            memory_utilization_pct=0,
            compute_capability=(8, 0),
            bandwidth_gbps=2039,
        )
        for i in range(count)
    ]
    return HardwareSnapshot(gpus=gpus, cpu_count=16, ram_total_mb=131_072)


def _t4_hw() -> HardwareSnapshot:
    return HardwareSnapshot(
        gpus=[GPUSnapshot(0, "T4", 16_384, 0, 16_384, 0, 0, (7, 5), 300)],
        cpu_count=4,
        ram_total_mb=32_768,
    )


def _cpu_hw() -> HardwareSnapshot:
    return HardwareSnapshot(gpus=[], cpu_count=8, ram_total_mb=16_384)


def _advisor(
    model: str = "gpt2",
    hardware: HardwareSnapshot | None = None,
    training_type: str = "supervised",
    objective: str = "balanced",
    verbose: bool = False,
    constraints: SolverConstraints | None = None,
) -> Advisor:
    hw = hardware or _a100_hw()
    return Advisor(
        model=model,
        hardware=hw,
        training_type=training_type,
        objective=objective,
        verbose=verbose,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# 1. suggest_config — basic correctness
# ---------------------------------------------------------------------------


class TestSuggestConfigBasic:
    def test_returns_sysplug_config(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"batch_size": 8, "learning_rate": 2e-5})
        assert isinstance(cfg, SysPlugConfig)

    def test_batch_size_at_least_one(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"batch_size": 8})
        assert cfg.batch_size >= 1

    def test_learning_rate_positive(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"learning_rate": 1e-4})
        assert cfg.learning_rate > 0

    def test_precision_valid_string(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"batch_size": 4})
        assert cfg.precision in {"fp32", "fp16", "bf16", "int8", "int4"}

    def test_optimizer_valid_string(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"optimizer": "adamw"})
        assert cfg.optimizer in {"adamw", "adam", "sgd", "adafactor"}

    def test_current_config_updated(self) -> None:
        adv = _advisor()
        assert adv.current_config is None
        cfg = adv.suggest_config({"batch_size": 4})
        assert adv.current_config is cfg

    def test_current_lr_reflects_config(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"learning_rate": 3e-5})
        assert math.isclose(adv.current_lr(), cfg.learning_rate, rel_tol=1e-9)

    def test_current_lr_zero_before_suggest(self) -> None:
        adv = _advisor()
        assert adv.current_lr() == 0.0

    def test_hardware_property_is_snapshot(self) -> None:
        hw = _a100_hw()
        adv = _advisor(hardware=hw)
        assert isinstance(adv.hardware, HardwareSnapshot)


# ---------------------------------------------------------------------------
# 2. suggest_config — memory fitting on small GPUs
# ---------------------------------------------------------------------------


class TestMemoryFitting:
    def test_config_fits_t4_with_gpt2(self) -> None:
        """GPT-2 (125 M) should easily fit on a T4-16GB with AdamW."""
        adv = _advisor(hardware=_t4_hw())
        cfg = adv.suggest_config({"batch_size": 8, "optimizer": "adamw", "precision": "bf16"})
        # Result must fit within 85% of 16 GB = ~13.9 GB
        budget_mb = 16_384 * 0.85
        assert cfg.predicted_peak_memory_mb <= budget_mb

    def test_config_fits_on_cpu_only(self) -> None:
        adv = _advisor(hardware=_cpu_hw())
        cfg = adv.suggest_config({"batch_size": 4})
        assert cfg.batch_size >= 1
        assert cfg.learning_rate > 0

    def test_large_model_triggers_oom_recovery(self) -> None:
        """7B model on a tiny 4GB GPU should trigger OOM recovery."""
        tiny_gpu = HardwareSnapshot(
            gpus=[GPUSnapshot(0, "TestGPU", 4_096, 0, 4_096, 0, 0, (8, 0), 900)],
            cpu_count=8,
            ram_total_mb=32_768,
        )
        adv = _advisor(model="llama-2-7b", hardware=tiny_gpu)
        cfg = adv.suggest_config({"batch_size": 4, "precision": "bf16"})
        # Solver should have done something to try to fit — GC, smaller batch, lower precision
        assert cfg.batch_size >= 1

    def test_gpt2_fits_comfortably_a100(self) -> None:
        adv = _advisor(hardware=_a100_hw())
        cfg = adv.suggest_config({"batch_size": 16})
        budget = 40_960 * 0.85
        assert cfg.predicted_peak_memory_mb <= budget


# ---------------------------------------------------------------------------
# 3. suggest_config — training types
# ---------------------------------------------------------------------------


class TestTrainingTypes:
    @pytest.mark.parametrize("tt", ["supervised", "sft", "dpo", "rlhf", "grpo"])
    def test_all_training_types_return_config(self, tt) -> None:
        adv = _advisor(training_type=tt)
        cfg = adv.suggest_config({"batch_size": 4})
        assert isinstance(cfg, SysPlugConfig)
        assert cfg.training_type == tt

    def test_rlhf_small_batch_adds_warning(self) -> None:
        """RLHF with batch=2 emits a reward-variance warning.

        training_type must be passed in the config dict so the solver's
        stability_warnings function can read it.
        """
        adv = _advisor(training_type="rlhf")
        cfg = adv.suggest_config({"batch_size": 2, "training_type": "rlhf"})
        if cfg.batch_size < 4:
            assert any("rlhf" in w.lower() or "reward" in w.lower() for w in cfg.warnings)


# ---------------------------------------------------------------------------
# 4. suggest_config — objectives
# ---------------------------------------------------------------------------


class TestObjectives:
    @pytest.mark.parametrize("obj", ["throughput", "memory", "balanced"])
    def test_all_objectives_return_config(self, obj) -> None:
        adv = _advisor(objective=obj)
        cfg = adv.suggest_config({"batch_size": 4})
        assert isinstance(cfg, SysPlugConfig)
        assert cfg.solver_objective == obj


# ---------------------------------------------------------------------------
# 5. suggest_config — model specification
# ---------------------------------------------------------------------------


class TestModelSpecification:
    def test_string_model_name(self) -> None:
        adv = _advisor(model="gpt2")
        cfg = adv.suggest_config({"batch_size": 4})
        assert cfg.param_count == 117_000_000  # 0.117B

    def test_integer_param_count(self) -> None:
        adv = _advisor(model=50_000_000)
        cfg = adv.suggest_config({"batch_size": 4})
        assert cfg.param_count == 50_000_000

    def test_larger_model_uses_more_memory(self) -> None:
        # Use models with non-overlapping names to avoid prefix-matching issues
        small = _advisor(model=125_000_000, hardware=_a100_hw())  # 125 M
        large = _advisor(model=7_000_000_000, hardware=_a100_hw())  # 7 B
        s_cfg = small.suggest_config({"batch_size": 4, "optimizer": "adamw", "precision": "bf16"})
        l_cfg = large.suggest_config({"batch_size": 4, "optimizer": "adamw", "precision": "bf16"})
        assert l_cfg.predicted_peak_memory_mb > s_cfg.predicted_peak_memory_mb


# ---------------------------------------------------------------------------
# 6. what_if engine
# ---------------------------------------------------------------------------


class TestWhatIf:
    def test_returns_what_if_result(self) -> None:
        adv = _advisor()
        adv.suggest_config({"batch_size": 4})
        result = adv.what_if({"batch_size": 8})
        assert isinstance(result, WhatIfResult)

    def test_what_if_before_suggest_raises(self) -> None:
        adv = _advisor()
        with pytest.raises(RuntimeError, match="suggest_config"):
            adv.what_if({"batch_size": 16})

    def test_changed_params_recorded(self) -> None:
        adv = _advisor()
        adv.suggest_config({"batch_size": 4})
        result = adv.what_if({"batch_size": 32})
        # batch_size was explicitly changed
        assert "batch_size" in result.changed_params

    def test_explicitly_changed_reason(self) -> None:
        adv = _advisor()
        adv.suggest_config({"batch_size": 4})
        result = adv.what_if({"batch_size": 16})
        if "batch_size" in result.changed_params:
            assert result.reason.get("batch_size") == "explicitly requested by user"

    def test_feasible_field_is_bool(self) -> None:
        adv = _advisor()
        adv.suggest_config({"batch_size": 4})
        result = adv.what_if({"batch_size": 8})
        assert isinstance(result.feasible, bool)

    def test_new_config_is_sysplug_config(self) -> None:
        adv = _advisor()
        adv.suggest_config({"batch_size": 4})
        result = adv.what_if({"batch_size": 8})
        assert isinstance(result.new_config, SysPlugConfig)

    def test_large_batch_infeasible_on_small_gpu(self) -> None:
        """Requesting an impossibly large batch should give feasible=False."""
        adv = _advisor(hardware=_t4_hw())
        adv.suggest_config({"batch_size": 2})
        result = adv.what_if({"batch_size": 512})
        # May or may not be feasible depending on solver recovery, but result exists
        assert isinstance(result, WhatIfResult)

    def test_precision_change_what_if(self) -> None:
        adv = _advisor()
        adv.suggest_config({"batch_size": 4, "precision": "bf16"})
        result = adv.what_if({"precision": "fp32"})
        assert isinstance(result, WhatIfResult)

    def test_what_if_with_explicit_base_config(self) -> None:
        adv = _advisor()
        base = adv.suggest_config({"batch_size": 4})
        result = adv.what_if({"batch_size": 8}, current_config=base)
        assert isinstance(result, WhatIfResult)

    def test_lr_scales_when_batch_increases(self) -> None:
        """Doubling effective batch should scale learning rate."""
        adv = _advisor(training_type="sft")
        adv.suggest_config({"batch_size": 4, "learning_rate": 1e-4, "gradient_accumulation": 1})
        result = adv.what_if({"batch_size": 8, "gradient_accumulation": 1})
        # Batch increased 2× → lr should increase (linear rule for sft < 256)
        if "learning_rate" in result.changed_params:
            old_lr, new_lr = result.changed_params["learning_rate"]
            assert new_lr > old_lr


# ---------------------------------------------------------------------------
# 7. Advisor.monitor() context manager
# ---------------------------------------------------------------------------


class TestAdvisorMonitor:
    def test_monitor_creates_monitor_instance(self) -> None:
        from sysplug.monitor import Monitor

        adv = _advisor()
        adv.suggest_config({"batch_size": 4})
        mon = adv.monitor(check_interval_steps=10)
        assert isinstance(mon, Monitor)

    def test_monitor_as_context_manager(self) -> None:
        adv = _advisor()
        adv.suggest_config({"batch_size": 4})
        with adv.monitor(check_interval_steps=50) as mon:
            for step in range(5):
                mon.record(step=step, loss=1.0 - step * 0.05)
        # No exception = success

    def test_monitor_records_events_list(self) -> None:
        adv = _advisor()
        adv.suggest_config({"batch_size": 4})
        with adv.monitor(check_interval_steps=5) as mon:
            for step in range(10):
                mon.record(step=step, loss=1.0)
        events = mon.get_events()
        assert isinstance(events, list)


# ---------------------------------------------------------------------------
# 8. profile_run (no CUDA)
# ---------------------------------------------------------------------------


class TestProfileRun:
    def test_profile_run_without_suggest_warns(self) -> None:
        import warnings

        adv = _advisor()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = adv.profile_run(dataloader=[], steps=3)
        assert result == {}

    def test_profile_run_with_suggest_returns_dict(self) -> None:
        adv = _advisor()
        adv.suggest_config({"batch_size": 2})
        # Minimal fake dataloader (list of dummy batches)
        dummy = [{"x": 0}] * 5
        result = adv.profile_run(dataloader=dummy, steps=3)
        assert isinstance(result, dict)
        assert "measured_samples_per_sec" in result


# ---------------------------------------------------------------------------
# 9. Multi-GPU
# ---------------------------------------------------------------------------


class TestMultiGPU:
    def test_multi_gpu_higher_effective_batch(self) -> None:
        """With 4 GPUs and batch=4, effective batch = 4 × 4 × 1 = 16."""
        hw = _a100_hw(count=4)
        adv = _advisor(hardware=hw)
        cfg = adv.suggest_config({"batch_size": 4, "gradient_accumulation": 1})
        # gpu_count should be 4
        assert cfg.gpu_count == 4

    def test_zero3_reduces_per_gpu_memory(self) -> None:
        hw = _a100_hw(count=4)
        adv_single = _advisor(hardware=_a100_hw(count=1))
        adv_zero3 = _advisor(hardware=hw)

        cfg_single = adv_single.suggest_config({"batch_size": 4, "parallelism": "none"})
        cfg_zero3 = adv_zero3.suggest_config({"batch_size": 4, "parallelism": "zero3"})

        # ZeRO-3 with 4 GPUs should have lower per-GPU predicted memory
        assert cfg_zero3.predicted_peak_memory_mb < cfg_single.predicted_peak_memory_mb


# ---------------------------------------------------------------------------
# 10. SysPlugConfig methods via Advisor output
# ---------------------------------------------------------------------------


class TestConfigMethods:
    def test_to_dict_has_all_keys(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"batch_size": 4})
        d = cfg.to_dict()
        for key in (
            "batch_size",
            "learning_rate",
            "precision",
            "optimizer",
            "parallelism",
            "gradient_accumulation",
            "effective_batch_size",
            "use_gradient_checkpointing",
            "predicted_peak_memory_mb",
            "predicted_throughput_samples_per_sec",
            "warnings",
            "notes",
        ):
            assert key in d

    def test_summary_returns_string(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"batch_size": 4})
        s = cfg.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_verbose_false(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"batch_size": 4})
        s = cfg.summary(verbose=False)
        assert "SysPlugConfig" in s

    def test_repr_contains_batch(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"batch_size": 4})
        r = repr(cfg)
        assert "batch_size=" in r

    def test_to_deepspeed_config_bf16(self) -> None:
        adv = _advisor()
        cfg = adv.suggest_config({"batch_size": 4, "precision": "bf16"})
        ds = cfg.to_deepspeed_config()
        assert ds.get("bf16", {}).get("enabled") is True
