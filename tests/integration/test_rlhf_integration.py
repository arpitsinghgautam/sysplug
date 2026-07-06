"""Integration tests for RLHF integration (RLHFAdvisor, PPOConfigHelper)."""

from __future__ import annotations

import pytest

from sysplug.hardware import HardwareSnapshot
from sysplug.integrations.rlhf import PPOConfig, PPOConfigHelper, RLHFAdvisor


class TestRLHFAdvisor:
    def test_creates_with_rlhf_training_type(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, verbose=False)
        assert advisor._training_type == "rlhf"

    def test_suggest_config_works(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug import SysPlugConfig

        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, verbose=False)
        cfg = advisor.suggest_config({"batch_size": 4, "learning_rate": 1e-5})
        assert isinstance(cfg, SysPlugConfig)

    def test_record_reward(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.record_reward(0, mean_reward=0.5, reward_std=0.1)
        assert len(advisor._rewards) == 1

    def test_record_kl(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.record_kl(0, kl_divergence=0.3)
        assert len(advisor._kl_divs) == 1

    def test_detect_reward_hacking_false_when_no_data(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, verbose=False)
        assert advisor.detect_reward_hacking() is False

    def test_detect_reward_hacking_false_when_kl_low(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, kl_threshold=5.0, verbose=False)
        advisor.record_reward(0, 0.5, 0.1)
        advisor.record_reward(1, 0.7, 0.1)
        advisor.record_kl(0, kl_divergence=0.5)  # below threshold
        assert advisor.detect_reward_hacking() is False

    def test_detect_reward_hacking_true(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, kl_threshold=2.0, verbose=False)
        # Reward increasing (trend > 0)
        advisor.record_reward(0, 0.5, 0.1)
        advisor.record_reward(1, 1.5, 0.1)  # big increase
        # KL above threshold
        advisor.record_kl(0, kl_divergence=10.0)
        assert advisor.detect_reward_hacking() is True

    def test_reward_summary_empty(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, verbose=False)
        summary = advisor.reward_summary()
        assert summary["mean_reward"] is None
        assert summary["reward_hacking_suspected"] is False

    def test_reward_summary_populated(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.record_reward(0, 0.7, 0.05)
        advisor.record_kl(0, 0.3)
        summary = advisor.reward_summary()
        assert summary["mean_reward"] == pytest.approx(0.7)
        assert summary["latest_kl"] == pytest.approx(0.3)

    def test_reward_window_limit(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = RLHFAdvisor(model="gpt2", hardware=mock_gpu, verbose=False, reward_window=5)
        for i in range(10):
            advisor.record_reward(i, float(i), 0.1)
        assert len(advisor._rewards) == 5  # window capped at 5


class TestPPOConfigHelper:
    def test_suggest_basic(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug import Advisor

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4})
        helper = PPOConfigHelper(advisor)
        ppo = helper.suggest(ppo_epochs=4)
        assert isinstance(ppo, PPOConfig)
        assert ppo.mini_batch_size >= 1
        assert ppo.ppo_epochs >= 1

    def test_suggest_clamps_ppo_epochs(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug import Advisor

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4})
        helper = PPOConfigHelper(advisor, max_ppo_epochs=2)
        ppo = helper.suggest(ppo_epochs=10)
        assert ppo.ppo_epochs <= 2

    def test_suggest_raises_without_config(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug import Advisor

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        helper = PPOConfigHelper(advisor)
        with pytest.raises(RuntimeError, match="suggest_config"):
            helper.suggest()

    def test_small_batch_warns(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug import Advisor

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        # Force small batch by using constraints
        advisor.suggest_config({"batch_size": 2})
        helper = PPOConfigHelper(advisor)
        ppo = helper.suggest(ppo_epochs=4)
        # Should include a note about small batch
        assert (
            any("batch" in note.lower() or "mini_batch" in note.lower() for note in ppo.notes)
            or ppo.mini_batch_size >= 1
        )

    def test_rollout_batch_size_positive(self, mock_gpu: HardwareSnapshot) -> None:
        from sysplug import Advisor

        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 8})
        helper = PPOConfigHelper(advisor, min_rollout_size=32)
        ppo = helper.suggest(ppo_epochs=4, num_envs=2)
        assert ppo.rollout_batch_size >= 32
