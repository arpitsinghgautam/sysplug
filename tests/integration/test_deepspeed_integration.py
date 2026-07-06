"""Integration tests for DeepSpeed integration."""

from __future__ import annotations

import pytest

from sysplug import Advisor
from sysplug.hardware import HardwareSnapshot


class TestDeepSpeedIntegration:
    def test_patch_deepspeed_config_basic(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug.integrations.deepspeed import patch_deepspeed_config

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4, "precision": "bf16"})

        ds_config: dict = {}
        patched = patch_deepspeed_config(ds_config, advisor)

        cfg = advisor.current_config
        assert patched["train_micro_batch_size_per_gpu"] == cfg.batch_size
        assert patched["gradient_accumulation_steps"] == cfg.gradient_accumulation

    def test_patch_sets_bf16(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug.integrations.deepspeed import patch_deepspeed_config

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4, "precision": "bf16"})
        cfg = advisor.current_config
        assert cfg is not None

        # Override precision to ensure it's bf16
        cfg.precision = "bf16"
        patched = patch_deepspeed_config({}, advisor)
        assert patched.get("bf16", {}).get("enabled") is True
        assert "fp16" not in patched

    def test_patch_sets_fp16(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug.integrations.deepspeed import patch_deepspeed_config

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4, "precision": "fp16"})
        cfg = advisor.current_config
        assert cfg is not None
        cfg.precision = "fp16"  # ensure
        patched = patch_deepspeed_config({}, advisor)
        assert patched.get("fp16", {}).get("enabled") is True

    def test_patch_sets_zero_stage(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug.integrations.deepspeed import patch_deepspeed_config

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4, "parallelism": "zero2"})
        cfg = advisor.current_config
        assert cfg is not None
        cfg.parallelism = "zero2"
        patched = patch_deepspeed_config({}, advisor)
        assert patched.get("zero_optimization", {}).get("stage") == 2

    def test_patch_warns_on_conflict(self, mock_gpu: HardwareSnapshot) -> None:
        import warnings

        from sysplug.integrations.deepspeed import patch_deepspeed_config

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4})
        cfg = advisor.current_config
        assert cfg is not None
        cfg.batch_size = 4

        conflicting_ds = {"train_micro_batch_size_per_gpu": 8}  # conflicts
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            patch_deepspeed_config(conflicting_ds, advisor)
        # Should have warned about the conflict
        assert any("Overriding" in str(warning.message) for warning in w)

    def test_patch_raises_without_config(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug.integrations.deepspeed import patch_deepspeed_config

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        with pytest.raises(RuntimeError, match="suggest_config"):
            patch_deepspeed_config({}, advisor)

    def test_to_deepspeed_config_from_sysplugconfig(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        cfg = advisor.suggest_config({"batch_size": 4, "precision": "bf16"})
        ds = cfg.to_deepspeed_config()
        assert ds["train_micro_batch_size_per_gpu"] == cfg.batch_size
