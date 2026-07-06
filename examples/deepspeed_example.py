"""DeepSpeed ZeRO-2 training example with SysPlug.

Demonstrates patch_deepspeed_config() integration.
Runs without DeepSpeed installed (prints config only).

Requires for full run: pip install sysplug[deepspeed]
"""

from __future__ import annotations

import json

import torch.nn as nn

import sysplug
from sysplug.integrations.deepspeed import patch_deepspeed_config


def main() -> None:
    print("=" * 60)
    print("SysPlug × DeepSpeed Example")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))

    advisor = sysplug.Advisor(
        model=model,
        training_type="sft",
        verbose=True,
    )
    cfg = advisor.suggest_config(
        {
            "batch_size": 4,
            "learning_rate": 2e-5,
            "precision": "bf16",
            "parallelism": "zero2",
        }
    )

    # Base DeepSpeed config (user-provided skeleton)
    base_ds_config: dict = {
        "zero_optimization": {
            "stage": 0,  # will be overridden
            "allgather_partitions": True,
            "reduce_scatter": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 50,
        "wall_clock_breakdown": False,
    }

    # Patch with SysPlug recommendations
    patched = patch_deepspeed_config(base_ds_config, advisor)

    print("\nPatched DeepSpeed config:")
    print(json.dumps(patched, indent=2))

    print("\nEquivalent TrainingArguments fields:")
    print(f"  per_device_train_batch_size = {cfg.batch_size}")
    print(f"  gradient_accumulation_steps = {cfg.gradient_accumulation}")
    print(f"  learning_rate               = {cfg.learning_rate:.2e}")
    print(f"  bf16                        = {cfg.precision == 'bf16'}")
    print(f"  gradient_checkpointing      = {cfg.use_gradient_checkpointing}")

    if cfg.warnings:
        print("\nWarnings:")
        for w in cfg.warnings:
            print(f"  [!] {w}")

    print("\n[DONE] DeepSpeed config patched successfully.")


if __name__ == "__main__":
    main()
