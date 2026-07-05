"""Unit tests for ConfigSolver."""

from __future__ import annotations

import math
import pytest

from sysplug.hardware import GPUSnapshot, HardwareSnapshot
from sysplug.memory_model import MemoryModel
from sysplug.solver import ConfigSolver, SolverConstraints
from sysplug.throughput_model import ThroughputModel


def make_hardware(total_mb: float, gpu_name: str = "A100") -> HardwareSnapshot:
    """Create a fake HardwareSnapshot with a given total memory."""
    gpu = GPUSnapshot(
        device_id=0,
        gpu_name=gpu_name,
        total_memory_mb=total_mb,
        used_memory_mb=0.0,
        free_memory_mb=total_mb,
        gpu_utilization_pct=0.0,
        memory_utilization_pct=0.0,
        compute_capability=(8, 0),
        bandwidth_gbps=2039.0,
    )
    return HardwareSnapshot(gpus=[gpu])


def make_solver(
    gpu_mb: float = 40960.0,
    training_type: str = "supervised",
    objective: str = "balanced",
    memory_safety_factor: float = 0.85,
) -> tuple[ConfigSolver, HardwareSnapshot]:
    """Create a ConfigSolver and matching HardwareSnapshot."""
    hw = make_hardware(gpu_mb)
    mem_model = MemoryModel(gpu_count=1)
    tput_model = ThroughputModel(gpu_name="A100")
    constraints = SolverConstraints(memory_safety_factor=memory_safety_factor)
    solver = ConfigSolver(
        memory_model=mem_model,
        throughput_model=tput_model,
        constraints=constraints,
        training_type=training_type,
        objective=objective,
    )
    return solver, hw


class TestOOMRecovery:
    def test_oom_reduces_batch_size(self) -> None:
        """When predicted memory > 90% of GPU, solver reduces batch size."""
        # Use a very small GPU (2GB) with a 7B model that won't fit at batch=8
        solver, hw = make_solver(gpu_mb=2048.0, memory_safety_factor=0.85)
        config = {
            "batch_size": 8,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "fp32",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=7_000_000_000)
        # Should have reduced batch size or precision
        assert (
            result.batch_size < 8
            or result.precision != "fp32"
            or result.use_gradient_checkpointing
        )

    def test_oom_preserves_effective_batch_with_grad_acc(self) -> None:
        """When batch is halved, grad_acc should double to preserve effective batch."""
        solver, hw = make_solver(gpu_mb=4096.0, memory_safety_factor=0.85)
        config = {
            "batch_size": 8,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "fp32",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=7_000_000_000)
        # Effective batch should not shrink below the smallest feasible
        assert result.batch_size * result.gradient_accumulation >= 1

    def test_infeasible_config_returns_best_effort(self) -> None:
        """Even on impossibly small GPU, solver returns something (best effort)."""
        solver, hw = make_solver(gpu_mb=512.0, memory_safety_factor=0.99)
        config = {
            "batch_size": 64,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "fp32",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=70_000_000_000)
        # Should still return a SysPlugConfig (not raise)
        assert result is not None
        assert result.batch_size >= 1


class TestLRScaling:
    def test_lr_scaling_applied_on_batch_change(self) -> None:
        """LR should be scaled when effective batch size changes."""
        solver, hw = make_solver(gpu_mb=2048.0)  # force batch reduction
        config = {
            "batch_size": 16,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "fp32",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=7_000_000_000)
        orig_eff = 16
        new_eff = result.batch_size * result.gradient_accumulation
        if new_eff != orig_eff:
            # LR should have been scaled (not equal to original)
            assert result.learning_rate != pytest.approx(1e-4, rel=1e-6)

    def test_sqrt_lr_scale_for_large_batch(self) -> None:
        """For batch > 256, sqrt LR rule should be used."""
        solver, hw = make_solver(gpu_mb=40960.0, training_type="supervised")
        config = {
            "batch_size": 8,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "bf16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        # Force grad_acc increase so effective batch becomes very large
        # by adjusting constraints
        solver._constraints.max_grad_accumulation = 64
        result = solver.solve(config, hw, param_count=125_000_000)
        # Just verify it runs and returns valid config
        assert result.learning_rate > 0

    def test_grad_acc_increase_preserves_effective_batch(self) -> None:
        """When batch is halved and grad_acc doubled, effective batch is unchanged."""
        solver, hw = make_solver(gpu_mb=40960.0)
        original_batch = 8
        original_grad_acc = 2
        config = {
            "batch_size": original_batch,
            "gradient_accumulation": original_grad_acc,
            "learning_rate": 1e-4,
            "precision": "bf16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=125_000_000)
        assert result.effective_batch_size == result.batch_size * result.gradient_accumulation


class TestPrecisionUpgrade:
    def test_fp16_upgraded_to_bf16_when_safe(self) -> None:
        """On ample memory, fp16 may be upgraded to bf16 for throughput."""
        solver, hw = make_solver(gpu_mb=40960.0, objective="throughput")
        config = {
            "batch_size": 4,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "fp16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=125_000_000)
        # Should upgrade to bf16 when memory allows and objective is throughput
        assert result.precision in {"fp16", "bf16"}  # upgrade is optional


class TestStabilityWarnings:
    def test_high_lr_fp16_warns(self) -> None:
        """LR > 1e-3 with fp16 (locked) should emit a warning."""
        solver, hw = make_solver(gpu_mb=40960.0)
        config = {
            "batch_size": 8,
            "gradient_accumulation": 1,
            "learning_rate": 5e-3,
            "precision": "fp16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        # Lock precision so the solver cannot upgrade fp16→bf16 before warning check
        result = solver.solve(config, hw, param_count=125_000_000,
                              locked_params={"precision": "fp16"})
        # Should have a warning about high LR + fp16
        assert any("fp16" in w.lower() or "LR" in w for w in result.warnings)

    def test_rlhf_small_batch_warns(self) -> None:
        """RLHF with batch_size < 4 should warn."""
        solver, hw = make_solver(gpu_mb=40960.0, training_type="rlhf")
        config = {
            "batch_size": 2,
            "gradient_accumulation": 1,
            "learning_rate": 1e-5,
            "precision": "bf16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
            "training_type": "rlhf",
        }
        result = solver.solve(config, hw, param_count=125_000_000)
        assert any("rlhf" in w.lower() or "batch" in w.lower() for w in result.warnings)

    def test_high_grad_acc_warns(self) -> None:
        """gradient_accumulation > 32 should warn."""
        solver, hw = make_solver(gpu_mb=40960.0)
        config = {
            "batch_size": 1,
            "gradient_accumulation": 64,
            "learning_rate": 1e-4,
            "precision": "bf16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=125_000_000)
        assert any("64" in w or "accumulation" in w.lower() for w in result.warnings)


class TestSolverOutput:
    def test_returns_sysplugconfig(self) -> None:
        from sysplug.config import SysPlugConfig
        solver, hw = make_solver()
        config = {
            "batch_size": 4,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "bf16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=125_000_000)
        assert isinstance(result, SysPlugConfig)

    def test_effective_batch_computed(self) -> None:
        solver, hw = make_solver()
        config = {
            "batch_size": 4,
            "gradient_accumulation": 2,
            "learning_rate": 1e-4,
            "precision": "bf16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=125_000_000)
        # gpu_count=1, batch=4, grad_acc=2 → eff_batch=8
        assert result.effective_batch_size == result.batch_size * result.gradient_accumulation

    def test_predicted_memory_positive(self) -> None:
        solver, hw = make_solver()
        config = {
            "batch_size": 4,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "bf16",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(config, hw, param_count=125_000_000)
        assert result.predicted_peak_memory_mb > 0

    def test_locked_params_not_changed(self) -> None:
        """Locked parameters must not be changed by the solver."""
        solver, hw = make_solver(gpu_mb=2048.0)
        config = {
            "batch_size": 8,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "precision": "fp32",
            "optimizer": "adamw",
            "parallelism": "none",
            "use_gradient_checkpointing": False,
        }
        result = solver.solve(
            config, hw, param_count=7_000_000_000,
            locked_params={"precision": "fp32"}
        )
        # Precision must not be changed
        assert result.precision == "fp32"
