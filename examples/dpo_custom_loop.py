"""Custom DPO training loop example with SysPlug monitoring.

Demonstrates a manual DPO-style training loop (without Trainer)
with SysPlug advisor and online monitoring.

Runs on CPU with tiny synthetic data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sysplug
from sysplug.integrations.pytorch import SysPlugContext


class PairDataset(Dataset):
    """Synthetic preference dataset: (chosen, rejected) pairs."""

    def __init__(self, n: int = 64, seq_len: int = 16, vocab: int = 50) -> None:
        self.chosen = torch.randint(0, vocab, (n, seq_len))
        self.rejected = torch.randint(0, vocab, (n, seq_len))

    def __len__(self) -> int:
        return len(self.chosen)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"chosen": self.chosen[idx], "rejected": self.rejected[idx]}


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 50, hidden: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x).mean(dim=1))

    def log_prob(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute mean log-probability for the token sequence."""
        logits = self.forward(tokens)
        return -F.cross_entropy(
            logits.unsqueeze(1).expand(-1, tokens.size(1), -1).reshape(-1, logits.size(-1)),
            tokens.reshape(-1),
            reduction="none",
        ).reshape(tokens.size(0), -1).mean(dim=-1)


def dpo_loss(
    policy: nn.Module,
    reference: nn.Module,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimisation loss (Rafailov et al. 2023)."""
    with torch.no_grad():
        ref_chosen = reference.log_prob(chosen)
        ref_rejected = reference.log_prob(rejected)

    pol_chosen = policy.log_prob(chosen)
    pol_rejected = policy.log_prob(rejected)

    logits = beta * ((pol_chosen - ref_chosen) - (pol_rejected - ref_rejected))
    return -F.logsigmoid(logits).mean()


def main() -> None:
    print("=" * 60)
    print("SysPlug DPO Custom Loop Example")
    print("=" * 60)

    policy = TinyLM()
    reference = TinyLM()
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad_(False)

    dataset = PairDataset(n=64)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    advisor = sysplug.Advisor(
        model=policy,
        training_type="dpo",
        objective="balanced",
        verbose=True,
    )
    cfg = advisor.suggest_config({
        "batch_size": 8,
        "learning_rate": 1e-4,
        "precision": "fp32",
    })

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)

    print(f"\nTraining with config: {cfg.summary(verbose=False)}")
    print()

    with SysPlugContext(advisor, check_interval_steps=5, reconfig_policy="warn-only") as ctx:
        for epoch in range(2):
            for step, batch in enumerate(loader):
                optimizer.zero_grad()
                loss = dpo_loss(policy, reference, batch["chosen"], batch["rejected"])
                loss.backward()
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in policy.parameters()
                    if p.grad is not None
                ) ** 0.5
                optimizer.step()

                global_step = epoch * len(loader) + step
                ctx.record(step=global_step, loss=loss.item(), grad_norm=grad_norm)

                if step % 4 == 0:
                    print(f"  epoch={epoch} step={step:3d}  loss={loss.item():.4f}")

    print("\n[DONE] DPO training complete.")


if __name__ == "__main__":
    main()
