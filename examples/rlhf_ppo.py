"""RLHF PPO example with SysPlug RLHFAdvisor.

Demonstrates:
- RLHFAdvisor with record_reward() / record_kl()
- detect_reward_hacking()
- PPOConfigHelper

Runs on CPU with a synthetic reward model.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from sysplug.integrations.rlhf import PPOConfigHelper, RLHFAdvisor


class TinyPolicy(nn.Module):
    def __init__(self, obs_dim: int = 16, act_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU(), nn.Linear(32, act_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fake_reward(actions: torch.Tensor) -> torch.Tensor:
    """Toy reward: higher for actions near zero."""
    return -actions.abs().mean(dim=-1)


def main() -> None:
    print("=" * 60)
    print("SysPlug RLHF/PPO Example")
    print("=" * 60)

    policy = TinyPolicy()
    advisor = RLHFAdvisor(
        model=policy,
        training_type="rlhf",
        objective="balanced",
        verbose=True,
        kl_threshold=3.0,
    )

    cfg = advisor.suggest_config(
        {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "precision": "fp32",
        }
    )
    print(f"\nRecommended config: {cfg.summary(verbose=False)}")

    # PPO batch configuration
    ppo_helper = PPOConfigHelper(advisor, min_rollout_size=32, max_ppo_epochs=4)
    ppo_cfg = ppo_helper.suggest(ppo_epochs=4, num_envs=1)
    print("\nPPO config:")
    print(f"  rollout_batch_size = {ppo_cfg.rollout_batch_size}")
    print(f"  mini_batch_size    = {ppo_cfg.mini_batch_size}")
    print(f"  ppo_epochs         = {ppo_cfg.ppo_epochs}")
    print(f"  learning_rate      = {ppo_cfg.learning_rate:.2e}")
    for note in ppo_cfg.notes:
        print(f"  NOTE: {note}")

    # Toy PPO training loop
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    rng = random.Random(0)

    print("\nRunning PPO steps...")
    for step in range(20):
        obs = torch.randn(ppo_cfg.mini_batch_size, 16)
        actions = policy(obs)
        rewards = fake_reward(actions)
        mean_reward = rewards.mean().item()
        reward_std = rewards.std().item()

        # Fake KL: gradually increasing
        kl = 0.1 * step + rng.uniform(-0.05, 0.05)

        # PPO loss (simplified)
        loss = -rewards.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        advisor.record_reward(step, mean_reward, reward_std)
        advisor.record_kl(step, kl)

        if step % 5 == 0:
            hacking = advisor.detect_reward_hacking()
            summary = advisor.reward_summary()
            print(f"  step={step:3d}  reward={mean_reward:.3f}  kl={kl:.2f}  hacking={hacking}")

    print("\n[DONE] RLHF/PPO example complete.")
    summary = advisor.reward_summary()
    print(f"Final reward summary: {summary}")


if __name__ == "__main__":
    main()
