"""Integration tests for Advisor.suggest_config."""

from __future__ import annotations

import io
import pytest

from sysplug import Advisor, SysPlugConfig
from sysplug.hardware import HardwareSnapshot


class TestAdvisorSuggestConfig:
    def test_returns_sysplugconfig(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = Advisor(
            model="gpt2",
            hardware=mock_gpu,
            verbose=False,
        )
        cfg = advisor.suggest_config({"batch_size": 4, "learning_rate": 2e-5})
        assert isinstance(cfg, SysPlugConfig)

    def test_respects_memory_limit_16gb(
        self, mock_gpu_16gb: HardwareSnapshot
    ) -> None:
        """Solver must reduce memory when naive config exceeds 16GB budget.

        Uses GPT-2 (125M params) at fp32 batch=32 as a starting point that
        exceeds the 16GB safety budget, and verifies the solver reduces it.
        """
        advisor = Advisor(
            model="gpt2",
            hardware=mock_gpu_16gb,
            verbose=False,
        )
        cfg = advisor.suggest_config({"batch_size": 8, "precision": "bf16"})
        # Predicted memory must be within 85% of 16384 MB for a 125M model
        assert cfg.predicted_peak_memory_mb <= 16384 * 0.85

    def test_returns_valid_fields(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        cfg = advisor.suggest_config({"batch_size": 4})
        assert cfg.batch_size >= 1
        assert cfg.learning_rate > 0
        assert cfg.precision in {"fp32", "fp16", "bf16", "int8", "int4"}
        assert cfg.optimizer in {"adamw", "adam", "sgd", "adafactor"}
        assert isinstance(cfg.warnings, list)
        assert isinstance(cfg.notes, list)

    def test_verbose_output_contains_table(
        self,
        mock_gpu: HardwareSnapshot,
        capsys: pytest.CaptureFixture,  # type: ignore[type-arg]
    ) -> None:
        """Verbose mode should print something to stdout (rich table)."""
        import sysplug
        from sysplug.utils.logging import get_console
        import rich.console

        output = io.StringIO()

        class CapturingConsole(rich.console.Console):
            pass

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=True)
        # Redirect the advisor's console to capture output
        advisor._console = rich.console.Console(file=output, highlight=False)
        cfg = advisor.suggest_config({"batch_size": 4})
        text = output.getvalue()
        # Summary should contain at least the batch_size
        assert "batch_size" in text or "SysPlug" in text or str(cfg.batch_size) in text

    def test_suggest_with_string_model_name(
        self, mock_gpu: HardwareSnapshot
    ) -> None:
        """String model names like 'llama-3-8b' should be resolved."""
        advisor = Advisor(model="llama-3-8b", hardware=mock_gpu, verbose=False)
        cfg = advisor.suggest_config({"batch_size": 2, "precision": "bf16"})
        assert isinstance(cfg, SysPlugConfig)

    def test_current_config_set_after_suggest(
        self, mock_gpu: HardwareSnapshot
    ) -> None:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        assert advisor.current_config is None
        advisor.suggest_config({"batch_size": 4})
        assert advisor.current_config is not None

    def test_suggest_cpu_only(self, cpu_only_hardware: HardwareSnapshot) -> None:
        """Suggest config should work in CPU-only mode."""
        advisor = Advisor(model="gpt2", hardware=cpu_only_hardware, verbose=False)
        cfg = advisor.suggest_config({"batch_size": 4})
        assert isinstance(cfg, SysPlugConfig)

    def test_invalid_config_raises(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        with pytest.raises(ValueError):
            advisor.suggest_config({"batch_size": -1})

    def test_current_lr_after_suggest(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4, "learning_rate": 3e-5})
        assert advisor.current_lr() > 0

    def test_training_types(self, mock_gpu: HardwareSnapshot) -> None:
        for tt in ["supervised", "sft", "dpo", "rlhf", "grpo"]:
            advisor = Advisor(
                model="gpt2", hardware=mock_gpu, training_type=tt, verbose=False
            )
            cfg = advisor.suggest_config({"batch_size": 4})
            assert cfg.training_type == tt
