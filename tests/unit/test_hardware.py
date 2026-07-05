"""Unit tests for HardwareProfiler."""

from __future__ import annotations

import pytest

from sysplug.hardware import (
    GPUSnapshot,
    HardwareProfiler,
    HardwareSnapshot,
    _estimate_bandwidth,
    _cpu_only_snapshot,
)


class TestBandwidthEstimation:
    @pytest.mark.parametrize("name,expected_min", [
        ("NVIDIA A100-SXM4-40GB", 1000.0),
        ("Tesla V100-SXM2-32GB", 500.0),
        ("Tesla T4", 200.0),
        ("NVIDIA RTX 4090", 500.0),
    ])
    def test_known_gpu_bandwidth(self, name: str, expected_min: float) -> None:
        bw = _estimate_bandwidth(name)
        assert bw >= expected_min

    def test_unknown_gpu_uses_default(self) -> None:
        bw = _estimate_bandwidth("XYZ-Unknown-9999")
        assert bw == 300.0  # default fallback


class TestGPUSnapshot:
    def test_snapshot_fields(self) -> None:
        gpu = GPUSnapshot(
            device_id=0,
            gpu_name="A100",
            total_memory_mb=40960.0,
            used_memory_mb=2048.0,
            free_memory_mb=38912.0,
            gpu_utilization_pct=50.0,
            memory_utilization_pct=10.0,
            compute_capability=(8, 0),
            bandwidth_gbps=2039.0,
        )
        assert gpu.device_id == 0
        assert gpu.total_memory_mb == 40960.0
        assert gpu.free_memory_mb == pytest.approx(38912.0)


class TestHardwareSnapshot:
    def test_gpu_count(self) -> None:
        gpu = GPUSnapshot(0, "A100", 40960, 0, 40960, 0, 0, (8, 0), 2039)
        snap = HardwareSnapshot(gpus=[gpu])
        assert snap.gpu_count == 1

    def test_empty_gpu_count(self) -> None:
        snap = HardwareSnapshot(gpus=[])
        assert snap.gpu_count == 0

    def test_total_memory_mb(self) -> None:
        gpu = GPUSnapshot(0, "A100", 40960, 0, 40960, 0, 0, (8, 0), 2039)
        snap = HardwareSnapshot(gpus=[gpu])
        assert snap.total_memory_mb(0) == 40960.0

    def test_total_memory_no_gpu(self) -> None:
        snap = HardwareSnapshot(gpus=[])
        assert snap.total_memory_mb(0) == 0.0

    def test_free_memory_mb(self) -> None:
        gpu = GPUSnapshot(0, "A100", 40960, 1024, 39936, 0, 0, (8, 0), 2039)
        snap = HardwareSnapshot(gpus=[gpu])
        assert snap.free_memory_mb(0) == 39936.0

    def test_min_free_memory_multi_gpu(self) -> None:
        gpus = [
            GPUSnapshot(0, "A100", 40960, 1000, 39960, 0, 0, (8, 0), 2039),
            GPUSnapshot(1, "A100", 40960, 5000, 35960, 0, 0, (8, 0), 2039),
        ]
        snap = HardwareSnapshot(gpus=gpus)
        assert snap.min_free_memory_mb() == pytest.approx(35960.0)

    def test_avg_utilization(self) -> None:
        gpus = [
            GPUSnapshot(0, "A100", 40960, 0, 40960, 60.0, 0, (8, 0), 2039),
            GPUSnapshot(1, "A100", 40960, 0, 40960, 80.0, 0, (8, 0), 2039),
        ]
        snap = HardwareSnapshot(gpus=gpus)
        assert snap.avg_utilization_pct() == pytest.approx(70.0)

    def test_is_cpu_only_flag(self) -> None:
        snap = HardwareSnapshot(gpus=[], is_cpu_only=True)
        assert snap.is_cpu_only is True


class TestHardwareProfilerCPUFallback:
    def test_no_nvml_returns_cpu_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When pynvml is unavailable, profiler returns CPU-only snapshot."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pynvml":
                raise ImportError("pynvml not found")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        profiler = HardwareProfiler(verbose=False)
        snap = profiler.snapshot()
        assert snap.is_cpu_only is True

    def test_snapshot_returns_hardware_snapshot(self) -> None:
        """snapshot() always returns a HardwareSnapshot."""
        profiler = HardwareProfiler(verbose=False)
        snap = profiler.snapshot()
        assert isinstance(snap, HardwareSnapshot)

    def test_cpu_only_snapshot_has_positive_cpu_count(self) -> None:
        snap = _cpu_only_snapshot()
        assert snap.cpu_count >= 1
        assert snap.is_cpu_only is True

    def test_poll_yields_snapshots(self) -> None:
        """poll() should yield at least one snapshot."""
        profiler = HardwareProfiler(verbose=False)
        gen = profiler.poll(interval_sec=0.01)
        snap = next(gen)
        assert isinstance(snap, HardwareSnapshot)
