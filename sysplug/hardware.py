"""Hardware profiling via pynvml.

Provides :class:`HardwareProfiler` which wraps NVML to expose GPU metrics
including memory, utilisation, and estimated memory bandwidth.  Falls back
gracefully to a CPU-only placeholder when no CUDA devices are available.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Generator, List, Optional

from sysplug.utils.logging import get_logger

log = get_logger()

# ---------------------------------------------------------------------------
# Bandwidth lookup: peak memory bandwidth in GB/s by GPU model name keywords.
# Sources: NVIDIA official spec sheets.
# ---------------------------------------------------------------------------
_BANDWIDTH_TABLE: dict[str, float] = {
    "A100": 2039.0,   # SXM4 80GB HBM2e
    "H100": 3350.0,   # SXM5 80GB HBM3
    "A10": 600.0,
    "A10G": 600.0,
    "A30": 933.0,
    "A40": 696.0,
    "A6000": 768.0,
    "V100": 900.0,    # SXM2
    "T4": 300.0,
    "P100": 732.0,
    "RTX 4090": 1008.0,
    "RTX 4080": 736.0,
    "RTX 3090": 936.0,
    "RTX 3080": 760.0,
    "RTX 3070": 448.0,
    "L40": 864.0,
    "L4": 300.0,
}

_DEFAULT_BANDWIDTH_GBPS = 300.0  # conservative fallback


def _estimate_bandwidth(gpu_name: str) -> float:
    """Return estimated memory bandwidth in GB/s from the GPU name string."""
    for keyword, bw in _BANDWIDTH_TABLE.items():
        if keyword.lower() in gpu_name.lower():
            return bw
    return _DEFAULT_BANDWIDTH_GBPS


@dataclass
class GPUSnapshot:
    """Metrics snapshot for a single GPU device.

    Attributes:
        device_id: CUDA device index (0-based).
        gpu_name: Human-readable GPU name string.
        total_memory_mb: Total VRAM in MiB.
        used_memory_mb: Currently allocated VRAM in MiB.
        free_memory_mb: Available VRAM in MiB.
        gpu_utilization_pct: SM utilisation percentage (0-100).
        memory_utilization_pct: Memory controller utilisation percentage (0-100).
        compute_capability: Tuple of (major, minor) compute capability.
        bandwidth_gbps: Estimated peak memory bandwidth in GB/s.
        temperature_c: GPU die temperature in Celsius (None if unavailable).
        power_draw_w: Current power draw in Watts (None if unavailable).
    """

    device_id: int
    gpu_name: str
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    gpu_utilization_pct: float
    memory_utilization_pct: float
    compute_capability: tuple[int, int]
    bandwidth_gbps: float
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None


@dataclass
class HardwareSnapshot:
    """System-wide hardware snapshot.

    Attributes:
        gpus: List of per-GPU snapshots, one per detected CUDA device.
        cpu_count: Number of logical CPU cores.
        ram_total_mb: Total system RAM in MiB.
        ram_available_mb: Available system RAM in MiB.
        timestamp: Unix timestamp when the snapshot was taken.
        is_cpu_only: True when no CUDA GPUs were found.
    """

    gpus: List[GPUSnapshot] = field(default_factory=list)
    cpu_count: int = 1
    ram_total_mb: float = 0.0
    ram_available_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)
    is_cpu_only: bool = False

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def gpu_count(self) -> int:
        """Number of detected CUDA devices."""
        return len(self.gpus)

    def total_memory_mb(self, device_id: int = 0) -> float:
        """Total VRAM in MiB for the given device."""
        return self.gpus[device_id].total_memory_mb if self.gpus else 0.0

    def free_memory_mb(self, device_id: int = 0) -> float:
        """Available VRAM in MiB for the given device."""
        return self.gpus[device_id].free_memory_mb if self.gpus else 0.0

    def min_free_memory_mb(self) -> float:
        """Minimum free VRAM across all GPUs (the binding constraint)."""
        if not self.gpus:
            return 0.0
        return min(g.free_memory_mb for g in self.gpus)

    def avg_utilization_pct(self) -> float:
        """Average SM utilisation across all GPUs."""
        if not self.gpus:
            return 0.0
        return sum(g.gpu_utilization_pct for g in self.gpus) / len(self.gpus)


def _cpu_only_snapshot() -> HardwareSnapshot:
    """Build a placeholder HardwareSnapshot for CPU-only environments."""
    try:
        import psutil  # type: ignore[import]
        cpu_count = psutil.cpu_count(logical=True) or 1
        mem = psutil.virtual_memory()
        ram_total_mb = mem.total / 1024 / 1024
        ram_available_mb = mem.available / 1024 / 1024
    except ImportError:
        cpu_count = 1
        ram_total_mb = 0.0
        ram_available_mb = 0.0

    return HardwareSnapshot(
        gpus=[],
        cpu_count=cpu_count,
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
        is_cpu_only=True,
    )


class HardwareProfiler:
    """Profiles available GPU hardware via pynvml.

    Uses NVIDIA Management Library (pynvml) to query live GPU metrics.
    Falls back to a CPU-only mode with a warning if pynvml is unavailable
    or no CUDA devices are present.

    Args:
        device_ids: List of GPU device IDs to monitor.  ``None`` means all.
        verbose: Emit warnings to the logger when running in CPU-only mode.

    Examples:
        >>> profiler = HardwareProfiler()
        >>> snap = profiler.snapshot()
        >>> snap.gpu_count
        0  # or however many GPUs are present
    """

    def __init__(
        self,
        device_ids: Optional[list[int]] = None,
        verbose: bool = True,
    ) -> None:
        self._device_ids = device_ids
        self._verbose = verbose
        self._nvml_available = False
        self._nvml_inited = False
        self._init_nvml()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_nvml(self) -> None:
        """Attempt to initialise pynvml; set cpu-only mode on failure."""
        try:
            import pynvml  # type: ignore[import]

            pynvml.nvmlInit()
            self._nvml_available = True
            self._nvml_inited = True
            self._pynvml = pynvml
        except Exception as exc:  # pynvml not installed or no CUDA
            if self._verbose:
                warnings.warn(
                    f"pynvml unavailable ({exc}); running in CPU-only mode. "
                    "Install pynvml and CUDA drivers for full GPU profiling.",
                    stacklevel=2,
                )
            self._nvml_available = False

    def _get_device_ids(self) -> list[int]:
        """Return the list of device IDs to query."""
        if not self._nvml_available:
            return []
        count = self._pynvml.nvmlDeviceGetCount()
        if self._device_ids is not None:
            return [d for d in self._device_ids if d < count]
        return list(range(count))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> HardwareSnapshot:
        """Capture a single point-in-time hardware snapshot.

        Returns:
            A :class:`HardwareSnapshot` with metrics for all requested GPUs,
            plus system CPU/RAM info.  If no GPUs are available,
            ``HardwareSnapshot.is_cpu_only`` is ``True``.

        Examples:
            >>> profiler = HardwareProfiler()
            >>> snap = profiler.snapshot()
            >>> isinstance(snap, HardwareSnapshot)
            True
        """
        if not self._nvml_available:
            return _cpu_only_snapshot()

        try:
            return self._build_snapshot()
        except Exception as exc:
            log.warning("Failed to read GPU metrics via pynvml: %s", exc)
            return _cpu_only_snapshot()

    def _build_snapshot(self) -> HardwareSnapshot:
        """Internal: query pynvml and build a HardwareSnapshot."""
        pynvml = self._pynvml
        gpu_snaps: list[GPUSnapshot] = []

        for dev_id in self._get_device_ids():
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = mem_info.total / 1024 / 1024
            used_mb = mem_info.used / 1024 / 1024
            free_mb = mem_info.free / 1024 / 1024

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = float(util.gpu)
                mem_util = float(util.memory)
            except pynvml.NVMLError:
                gpu_util = 0.0
                mem_util = 0.0

            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_cap: tuple[int, int] = (int(major), int(minor))
            except pynvml.NVMLError:
                compute_cap = (0, 0)

            try:
                temp: Optional[float] = float(
                    pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except pynvml.NVMLError:
                temp = None

            try:
                power: Optional[float] = float(
                    pynvml.nvmlDeviceGetPowerUsage(handle)
                ) / 1000.0  # mW -> W
            except pynvml.NVMLError:
                power = None

            gpu_snaps.append(
                GPUSnapshot(
                    device_id=dev_id,
                    gpu_name=name,
                    total_memory_mb=total_mb,
                    used_memory_mb=used_mb,
                    free_memory_mb=free_mb,
                    gpu_utilization_pct=gpu_util,
                    memory_utilization_pct=mem_util,
                    compute_capability=compute_cap,
                    bandwidth_gbps=_estimate_bandwidth(name),
                    temperature_c=temp,
                    power_draw_w=power,
                )
            )

        cpu_count = 1
        ram_total_mb = 0.0
        ram_available_mb = 0.0
        try:
            import psutil  # type: ignore[import]
            cpu_count = psutil.cpu_count(logical=True) or 1
            mem = psutil.virtual_memory()
            ram_total_mb = mem.total / 1024 / 1024
            ram_available_mb = mem.available / 1024 / 1024
        except ImportError:
            pass

        return HardwareSnapshot(
            gpus=gpu_snaps,
            cpu_count=cpu_count,
            ram_total_mb=ram_total_mb,
            ram_available_mb=ram_available_mb,
            is_cpu_only=len(gpu_snaps) == 0,
        )

    def poll(
        self, interval_sec: float = 1.0
    ) -> Generator[HardwareSnapshot, None, None]:
        """Yield hardware snapshots at a fixed polling interval forever.

        Args:
            interval_sec: Seconds to sleep between snapshots.

        Yields:
            Successive :class:`HardwareSnapshot` instances.

        Examples:
            >>> profiler = HardwareProfiler()
            >>> for snap in profiler.poll(interval_sec=0.5):
            ...     print(snap.avg_utilization_pct())
            ...     break  # stop after first snapshot
        """
        while True:
            yield self.snapshot()
            time.sleep(interval_sec)

    def __del__(self) -> None:
        """Shut down pynvml when the profiler is garbage-collected."""
        if self._nvml_inited and self._nvml_available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
