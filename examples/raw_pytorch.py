"""Raw PyTorch training loop example with SysPlug.

Demonstrates:
- advisor.suggest_config()
- advisor.what_if({"batch_size": 64})
- advisor.monitor(reconfig_policy="suggest")

Runs end-to-end on CPU with a tiny model and synthetic data.
No GPU required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sysplug
from sysplug.integrations.pytorch import SysPlugContext


def build_tiny_model(input_dim: int = 64, hidden: int = 128, output_dim: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )


def main() -> None:
    print("=" * 60)
    print("SysPlug Raw PyTorch Example")
    print("=" * 60)

    # Build model and data
    model = build_tiny_model()
    N, D = 256, 64
    dataset = TensorDataset(torch.randn(N, D), torch.randint(0, 10, (N,)))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # ----------------------------------------------------------------
    # Step 1: Get a recommended configuration
    # ----------------------------------------------------------------
    advisor = sysplug.Advisor(
        model=model,
        training_type="supervised",
        objective="balanced",
        verbose=True,
    )

    cfg = advisor.suggest_config({
        "batch_size": 8,
        "learning_rate": 1e-3,
        "precision": "fp32",
        "optimizer": "adamw",
        "gradient_accumulation": 1,
    })

    print(f"\nRecommended config: {cfg}")

    # ----------------------------------------------------------------
    # Step 2: What-if analysis
    # ----------------------------------------------------------------
    print("\n--- What-if: batch_size=32 ---")
    result = advisor.what_if({"batch_size": 32})
    print(f"Feasible: {result.feasible}")
    print(f"New config: {result.new_config}")
    if result.changed_params:
        print("Changes:")
        for k, (old, new) in result.changed_params.items():
            print(f"  {k}: {old!r} -> {new!r}  ({result.reason.get(k, '')})")

    # ----------------------------------------------------------------
    # Step 3: Training loop with monitoring
    # ----------------------------------------------------------------
    print("\n--- Training with online monitoring ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    with SysPlugContext(advisor, check_interval_steps=10, reconfig_policy="suggest") as ctx:
        for epoch in range(2):
            for step, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                global_step = epoch * len(loader) + step
                ctx.record(
                    step=global_step,
                    loss=loss.item(),
                    grad_norm=sum(
                        p.grad.norm().item() ** 2
                        for p in model.parameters()
                        if p.grad is not None
                    ) ** 0.5,
                )

    print("\n[DONE] Training completed. SysPlug monitoring finished.")

    # ----------------------------------------------------------------
    # Step 4: Show config as DeepSpeed config
    # ----------------------------------------------------------------
    ds_config = cfg.to_deepspeed_config()
    print(f"\nDeepSpeed config snippet: {ds_config}")


if __name__ == "__main__":
    main()
