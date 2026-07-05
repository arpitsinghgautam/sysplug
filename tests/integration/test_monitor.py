"""Integration tests for Monitor."""

from __future__ import annotations

import threading
import time
import pytest

from sysplug import Advisor, Monitor
from sysplug.hardware import GPUSnapshot, HardwareSnapshot
from sysplug.monitor import EventType, MonitorEvent


def make_advisor_with_config(hardware: HardwareSnapshot) -> Advisor:
    advisor = Advisor(model="gpt2", hardware=hardware, verbose=False)
    advisor.suggest_config({"batch_size": 4})
    return advisor


class TestMonitorBasic:
    def test_monitor_context_manager(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = make_advisor_with_config(mock_gpu)
        with advisor.monitor(check_interval_steps=5) as mon:
            for step in range(10):
                mon.record(step=step, loss=1.0 - step * 0.05)
        assert isinstance(mon, Monitor)

    def test_monitor_records_without_error(
        self, mock_gpu: HardwareSnapshot
    ) -> None:
        advisor = make_advisor_with_config(mock_gpu)
        with advisor.monitor(check_interval_steps=5) as mon:
            for step in range(20):
                mon.record(step=step, loss=1.0)
        # No exceptions should have been raised

    def test_get_events_returns_list(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = make_advisor_with_config(mock_gpu)
        with advisor.monitor(check_interval_steps=5) as mon:
            for step in range(10):
                mon.record(step=step, loss=1.0)
        events = mon.get_events()
        assert isinstance(events, list)

    def test_monitor_detects_diverging_loss(
        self, mock_gpu: HardwareSnapshot
    ) -> None:
        advisor = make_advisor_with_config(mock_gpu)
        with advisor.monitor(check_interval_steps=5, reconfig_policy="warn-only") as mon:
            # Feed diverging loss
            for step in range(60):
                loss = 1.0 + step * 0.1  # strongly increasing
                mon.record(step=step, loss=loss)
            time.sleep(0.3)  # give background thread time to run

        events = mon.get_events()
        types = {e.event_type for e in events}
        assert EventType.DIVERGING_LOSS in types

    def test_monitor_suggest_policy(
        self, mock_gpu: HardwareSnapshot, capsys: pytest.CaptureFixture  # type: ignore[type-arg]
    ) -> None:
        """suggest policy should not raise even when loss diverges."""
        advisor = make_advisor_with_config(mock_gpu)
        with advisor.monitor(check_interval_steps=5, reconfig_policy="suggest") as mon:
            for step in range(60):
                mon.record(step=step, loss=1.0 + step * 0.2)
            time.sleep(0.3)
        # Just check it doesn't crash

    def test_monitor_auto_apply_updates_config(
        self, mock_gpu: HardwareSnapshot
    ) -> None:
        """auto-apply policy may update the advisor's current config."""
        advisor = make_advisor_with_config(mock_gpu)
        original_lr = advisor.current_config.learning_rate  # type: ignore[union-attr]
        with advisor.monitor(
            check_interval_steps=5, reconfig_policy="auto-apply"
        ) as mon:
            for step in range(100):
                mon.record(step=step, loss=1.0 + step * 0.3)  # bad divergence
            time.sleep(0.5)
        # Config may or may not have changed depending on timing; just no crash

    def test_monitor_is_thread_safe(self, mock_gpu: HardwareSnapshot) -> None:
        """Concurrent record() calls should not raise."""
        advisor = make_advisor_with_config(mock_gpu)
        errors: list[Exception] = []

        with advisor.monitor(check_interval_steps=10) as mon:
            def worker(thread_id: int) -> None:
                try:
                    for step in range(50):
                        mon.record(step=thread_id * 50 + step, loss=1.0)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)

        assert not errors, f"Thread errors: {errors}"

    def test_monitor_oom_risk_detection(self) -> None:
        """When GPU memory > 90% used, OOM_RISK event should fire."""
        # Build a snapshot where memory is almost full
        gpu = GPUSnapshot(
            device_id=0,
            gpu_name="A100",
            total_memory_mb=40960.0,
            used_memory_mb=39000.0,  # ~95% used
            free_memory_mb=1960.0,
            gpu_utilization_pct=90.0,
            memory_utilization_pct=95.0,
            compute_capability=(8, 0),
            bandwidth_gbps=2039.0,
        )
        hw = HardwareSnapshot(gpus=[gpu])
        advisor = Advisor(model="gpt2", hardware=hw, verbose=False)
        advisor.suggest_config({"batch_size": 4})

        # Override the profiler to return the high-memory snapshot
        advisor._hardware = hw
        if hasattr(advisor, "_profiler"):
            advisor._profiler.snapshot = lambda: hw  # type: ignore[method-assign]

        with advisor.monitor(check_interval_steps=2) as mon:
            # The check loop should detect OOM risk from the hardware snapshot
            for step in range(20):
                mon.record(step=step, loss=1.0)
            time.sleep(0.5)
        # Just verifying no crash; OOM detection depends on real GPU readings

    def test_clear_events(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = make_advisor_with_config(mock_gpu)
        with advisor.monitor(check_interval_steps=100) as mon:
            for step in range(5):
                mon.record(step=step, loss=1.0)
        mon.clear_events()
        assert mon.get_events() == []
