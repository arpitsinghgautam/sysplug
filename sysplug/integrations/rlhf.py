"""RLHF / DPO / PPO training helpers for SysPlug.

Provides :class:`RLHFAdvisor` (a subclass of :class:`~sysplug.advisor.Advisor`)
with additional signals for reward hacking detection, and
:class:`PPOConfigHelper` for PPO-specific batch configuration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from sysplug.advisor import Advisor


@dataclass
class PPOConfig:
    """Recommended PPO training configuration.

    Attributes:
        rollout_batch_size: Total rollout batch size (env steps per update).
        ppo_epochs: Number of PPO update epochs per rollout.
        mini_batch_size: Mini-batch size for each gradient update.
        gradient_accumulation: Gradient accumulation steps.
        learning_rate: Recommended learning rate.
        notes: Explanatory notes.
    """

    rollout_batch_size: int
    ppo_epochs: int
    mini_batch_size: int
    gradient_accumulation: int
    learning_rate: float
    notes: list[str]


class RLHFAdvisor(Advisor):
    """Advisor subclass with RLHF-specific stability signals.

    Adds methods for recording reward statistics, KL divergence, and
    detecting reward hacking (when reward increases but KL diverges).

    Args:
        *args: Forwarded to :class:`~sysplug.advisor.Advisor`.
        reward_window: Number of steps in the reward sliding window.
        kl_threshold: KL divergence value above which reward hacking is flagged.
        **kwargs: Forwarded to :class:`~sysplug.advisor.Advisor`.

    Examples:
        >>> advisor = RLHFAdvisor(model="gpt2", training_type="rlhf")
        >>> advisor.record_reward(0, mean_reward=0.5, reward_std=0.1)
        >>> advisor.detect_reward_hacking()
        False
    """

    def __init__(
        self,
        *args: Any,
        reward_window: int = 20,
        kl_threshold: float = 5.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("training_type", "rlhf")
        super().__init__(*args, **kwargs)
        self._reward_window = reward_window
        self._kl_threshold = kl_threshold
        self._rewards: deque[tuple[int, float, float]] = deque(maxlen=reward_window)
        self._kl_divs: deque[tuple[int, float]] = deque(maxlen=reward_window)

    def record_reward(self, step: int, mean_reward: float, reward_std: float = 0.0) -> None:
        """Record reward statistics at a training step.

        Args:
            step: Training step index.
            mean_reward: Mean reward over the rollout batch.
            reward_std: Standard deviation of rewards in the batch.
        """
        self._rewards.append((step, mean_reward, reward_std))

    def record_kl(self, step: int, kl_divergence: float) -> None:
        """Record KL divergence from the reference policy.

        Args:
            step: Training step index.
            kl_divergence: KL divergence estimate.
        """
        self._kl_divs.append((step, kl_divergence))

    def detect_reward_hacking(self) -> bool:
        """Detect potential reward hacking.

        Returns ``True`` when:
        - Reward is trending upward, AND
        - KL divergence is above the configured threshold.

        Returns:
            ``True`` if reward hacking is suspected.

        Examples:
            >>> advisor = RLHFAdvisor(model="gpt2", kl_threshold=2.0)
            >>> advisor.record_reward(0, mean_reward=1.0, reward_std=0.1)
            >>> advisor.record_kl(0, kl_divergence=10.0)
            >>> advisor.detect_reward_hacking()
            True
        """
        if not self._rewards or not self._kl_divs:
            return False

        # Check if KL is too high
        latest_kl = self._kl_divs[-1][1]
        if latest_kl < self._kl_threshold:
            return False

        # Check if reward is trending upward (possible gaming)
        if len(self._rewards) < 2:
            return False
        rewards = [r for _, r, _ in self._rewards]
        trend = (rewards[-1] - rewards[0]) / (len(rewards) - 1)

        return trend > 0 and latest_kl >= self._kl_threshold

    def reward_summary(self) -> dict[str, Any]:
        """Return a summary of recorded reward and KL statistics.

        Returns:
            Dict with keys ``mean_reward``, ``reward_std``, ``latest_kl``,
            ``reward_hacking_suspected``.
        """
        if not self._rewards:
            return {
                "mean_reward": None,
                "reward_std": None,
                "latest_kl": None,
                "reward_hacking_suspected": False,
            }
        latest_reward = self._rewards[-1]
        latest_kl = self._kl_divs[-1][1] if self._kl_divs else None
        return {
            "mean_reward": latest_reward[1],
            "reward_std": latest_reward[2],
            "latest_kl": latest_kl,
            "reward_hacking_suspected": self.detect_reward_hacking(),
        }


class PPOConfigHelper:
    """Suggests PPO-specific batch configuration from GPU memory constraints.

    Args:
        advisor: A configured :class:`~sysplug.advisor.Advisor`.
        min_rollout_size: Minimum acceptable rollout batch size.
        max_ppo_epochs: Maximum PPO epochs per rollout update.

    Examples:
        >>> import sysplug
        >>> advisor = sysplug.Advisor(model="gpt2", training_type="rlhf")
        >>> _ = advisor.suggest_config({"batch_size": 4})
        >>> helper = PPOConfigHelper(advisor)
        >>> config = helper.suggest()
        >>> config.mini_batch_size >= 1
        True
    """

    def __init__(
        self,
        advisor: Advisor,
        min_rollout_size: int = 16,
        max_ppo_epochs: int = 4,
    ) -> None:
        self._advisor = advisor
        self._min_rollout_size = min_rollout_size
        self._max_ppo_epochs = max_ppo_epochs

    def suggest(self, ppo_epochs: int = 4, num_envs: int = 1) -> PPOConfig:
        """Suggest a PPO batch configuration within GPU memory limits.

        Algorithm:
        1. Use ``advisor.current_config.batch_size`` as the per-device mini-batch.
        2. Set rollout_batch_size = num_envs × mini_batch_size × ppo_epochs
           (clamped to at least ``min_rollout_size``).
        3. Scale learning_rate for RLHF regime.

        Args:
            ppo_epochs: Desired number of PPO update epochs per rollout.
            num_envs: Number of parallel rollout environments.

        Returns:
            A :class:`PPOConfig` with the recommended values.

        Raises:
            RuntimeError: If the advisor has no current config.
        """
        cfg = self._advisor.current_config
        if cfg is None:
            raise RuntimeError("Call advisor.suggest_config() first.")

        mini_batch = cfg.batch_size
        rollout_size = max(self._min_rollout_size, num_envs * mini_batch * ppo_epochs)
        # Clamp ppo_epochs to maximum
        ppo_epochs = min(ppo_epochs, self._max_ppo_epochs)

        # Gradient accumulation to handle large rollout
        grad_acc = max(1, rollout_size // (mini_batch * num_envs))

        notes: list[str] = []
        if mini_batch < 4:
            notes.append(
                f"mini_batch_size={mini_batch} is small for PPO; "
                "reward variance may be high. Consider using ≥4."
            )
        if ppo_epochs > 2 and rollout_size < 64:
            notes.append(
                "High ppo_epochs with small rollout may overfit policy; "
                "consider reducing ppo_epochs."
            )

        return PPOConfig(
            rollout_batch_size=rollout_size,
            ppo_epochs=ppo_epochs,
            mini_batch_size=mini_batch,
            gradient_accumulation=grad_acc,
            learning_rate=cfg.learning_rate,
            notes=notes,
        )
