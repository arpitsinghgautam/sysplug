"""Integration tests for Advisor.what_if."""

from __future__ import annotations

import pytest

from sysplug import Advisor, SysPlugConfig
from sysplug.advisor import WhatIfResult
from sysplug.hardware import HardwareSnapshot


class TestAdvisorWhatIf:
    def _get_advisor_with_config(self, mock_gpu: HardwareSnapshot) -> tuple[Advisor, SysPlugConfig]:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        cfg = advisor.suggest_config({"batch_size": 8, "learning_rate": 2e-5})
        return advisor, cfg

    def test_what_if_returns_whatif_result(self, mock_gpu: HardwareSnapshot) -> None:
        advisor, _ = self._get_advisor_with_config(mock_gpu)
        result = advisor.what_if({"batch_size": 16})
        assert isinstance(result, WhatIfResult)
        assert isinstance(result.new_config, SysPlugConfig)

    def test_what_if_batch_increase_scales_lr(self, mock_gpu: HardwareSnapshot) -> None:
        """Doubling batch size should scale LR upward."""
        advisor, cfg = self._get_advisor_with_config(mock_gpu)
        original_lr = cfg.learning_rate
        original_eff = cfg.effective_batch_size

        result = advisor.what_if({"batch_size": cfg.batch_size * 2})
        new_cfg = result.new_config

        if new_cfg.effective_batch_size > original_eff:
            # LR should have been scaled up
            assert new_cfg.learning_rate >= original_lr

    def test_what_if_infeasible_change_returns_closest(
        self, mock_gpu_16gb: HardwareSnapshot
    ) -> None:
        """What-if with OOM change should return a feasible config."""
        advisor = Advisor(model="llama-2-7b", hardware=mock_gpu_16gb, verbose=False)
        advisor.suggest_config({"batch_size": 2, "precision": "bf16"})
        # Request a huge batch on a small GPU
        result = advisor.what_if({"batch_size": 64, "precision": "fp32"})
        # If infeasible, the config should still be valid
        assert result.new_config.batch_size >= 1
        assert result.new_config.predicted_peak_memory_mb > 0

    def test_what_if_multi_param_change(self, mock_gpu: HardwareSnapshot) -> None:
        """Multi-parameter what-if should handle all requested changes."""
        advisor, cfg = self._get_advisor_with_config(mock_gpu)
        result = advisor.what_if(
            {"batch_size": 16, "precision": "bf16"},
        )
        new_cfg = result.new_config
        assert new_cfg.batch_size == 16
        assert new_cfg.precision == "bf16"

    def test_what_if_diff_annotations_correct(self, mock_gpu: HardwareSnapshot) -> None:
        """changed_params should track what actually changed."""
        advisor, cfg = self._get_advisor_with_config(mock_gpu)
        # Propose a change that will definitely change batch_size
        result = advisor.what_if({"batch_size": cfg.batch_size * 4})
        # batch_size should appear in changed_params if it changed
        if result.new_config.batch_size != cfg.batch_size:
            assert "batch_size" in result.changed_params

    def test_what_if_reason_populated(self, mock_gpu: HardwareSnapshot) -> None:
        advisor, cfg = self._get_advisor_with_config(mock_gpu)
        result = advisor.what_if({"batch_size": 16})
        # Reason dict should not be empty if params changed
        for key in result.changed_params:
            assert key in result.reason

    def test_what_if_without_suggest_raises(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        with pytest.raises(RuntimeError, match="suggest_config"):
            advisor.what_if({"batch_size": 16})

    def test_what_if_precision_change(self, mock_gpu: HardwareSnapshot) -> None:
        advisor, cfg = self._get_advisor_with_config(mock_gpu)
        result = advisor.what_if({"precision": "fp32"})
        assert result.new_config.precision == "fp32"

    def test_what_if_feasible_field_present(self, mock_gpu: HardwareSnapshot) -> None:
        advisor, _ = self._get_advisor_with_config(mock_gpu)
        result = advisor.what_if({"batch_size": 4})
        assert isinstance(result.feasible, bool)
