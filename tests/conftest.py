"""Shared pytest fixtures for SysPlug tests."""

from __future__ import annotations

import pytest

from sysplug.config import SysPlugConfig
from sysplug.hardware import GPUSnapshot, HardwareSnapshot


@pytest.fixture
def mock_gpu() -> HardwareSnapshot:
    """Fake HardwareSnapshot with A100-40GB specs (no real GPU required)."""
    gpu = GPUSnapshot(
        device_id=0,
        gpu_name="NVIDIA A100-SXM4-40GB",
        total_memory_mb=40960.0,  # 40 GiB
        used_memory_mb=512.0,
        free_memory_mb=40448.0,
        gpu_utilization_pct=0.0,
        memory_utilization_pct=0.0,
        compute_capability=(8, 0),
        bandwidth_gbps=2039.0,
        temperature_c=40.0,
        power_draw_w=250.0,
    )
    return HardwareSnapshot(
        gpus=[gpu],
        cpu_count=32,
        ram_total_mb=512 * 1024.0,
        ram_available_mb=400 * 1024.0,
        is_cpu_only=False,
    )


@pytest.fixture
def mock_gpu_16gb() -> HardwareSnapshot:
    """Fake HardwareSnapshot with 16GB GPU (e.g. T4/V100-16GB)."""
    gpu = GPUSnapshot(
        device_id=0,
        gpu_name="Tesla T4",
        total_memory_mb=16384.0,
        used_memory_mb=256.0,
        free_memory_mb=16128.0,
        gpu_utilization_pct=0.0,
        memory_utilization_pct=0.0,
        compute_capability=(7, 5),
        bandwidth_gbps=300.0,
        temperature_c=35.0,
        power_draw_w=70.0,
    )
    return HardwareSnapshot(
        gpus=[gpu],
        cpu_count=4,
        ram_total_mb=64 * 1024.0,
        ram_available_mb=50 * 1024.0,
        is_cpu_only=False,
    )


@pytest.fixture
def cpu_only_hardware() -> HardwareSnapshot:
    """Fake CPU-only HardwareSnapshot."""
    return HardwareSnapshot(
        gpus=[],
        cpu_count=8,
        ram_total_mb=32 * 1024.0,
        ram_available_mb=20 * 1024.0,
        is_cpu_only=True,
    )


@pytest.fixture
def tiny_model_param_count() -> int:
    """Known parameter count for a tiny 2-layer MLP."""
    # 2-layer MLP: input_dim=64, hidden=32, output=10
    # Layer 1: 64*32 + 32 = 2080 params
    # Layer 2: 32*10 + 10 = 330 params
    # Total: 2410 params
    return 2410


@pytest.fixture
def sample_config() -> dict:
    """Standard starting configuration dict."""
    return {
        "batch_size": 8,
        "gradient_accumulation": 1,
        "learning_rate": 2e-5,
        "precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "none",
        "use_gradient_checkpointing": False,
    }


@pytest.fixture
def sample_sysplug_config() -> SysPlugConfig:
    """A pre-built SysPlugConfig for use in what_if tests."""
    return SysPlugConfig(
        batch_size=8,
        gradient_accumulation=1,
        effective_batch_size=8,
        learning_rate=2e-5,
        precision="bf16",
        optimizer="adamw",
        parallelism="none",
        use_gradient_checkpointing=False,
        predicted_peak_memory_mb=5000.0,
        predicted_throughput_samples_per_sec=50.0,
        safety_margin_pct=0.85,
        training_type="supervised",
        gpu_count=1,
        param_count=125_000_000,
    )
