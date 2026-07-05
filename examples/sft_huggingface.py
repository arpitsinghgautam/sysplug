"""SFT training with Hugging Face Trainer + SysPlugTrainerCallback.

Demonstrates:
- advisor = sysplug.Advisor(model=model, training_type="sft")
- cfg = advisor.suggest_config(vars(training_args))
- training_args = cfg.to_training_arguments()
- trainer = Trainer(..., callbacks=[SysPlugTrainerCallback(advisor)])

Requires: pip install sysplug[hf]
Uses GPT-2 and a tiny synthetic dataset so it runs without a GPU.
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch.utils.data import Dataset


def main() -> None:
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
    except ImportError:
        print("transformers not installed. Run: pip install sysplug[hf]")
        return

    import sysplug
    from sysplug.integrations.huggingface import SysPlugTrainerCallback

    print("=" * 60)
    print("SysPlug × Hugging Face SFT Example")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Tiny synthetic dataset
    # ----------------------------------------------------------------
    class TinyTextDataset(Dataset):
        def __init__(self, tokenizer: object, n: int = 64) -> None:
            self.samples = [
                tokenizer(
                    "The quick brown fox jumps over the lazy dog. " * 3,
                    max_length=64,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                for _ in range(n)
            ]

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            item = self.samples[idx]
            return {k: v.squeeze(0) for k, v in item.items()}

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    train_dataset = TinyTextDataset(tokenizer, n=32)

    # ----------------------------------------------------------------
    # SysPlug advisor
    # ----------------------------------------------------------------
    advisor = sysplug.Advisor(model=model, training_type="sft", verbose=True)

    with tempfile.TemporaryDirectory() as tmp:
        base_args = TrainingArguments(
            output_dir=tmp,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            logging_steps=10,
            save_steps=500,
            no_cuda=not torch.cuda.is_available(),
            use_cpu=not torch.cuda.is_available(),
            report_to=[],
        )

        # Ask SysPlug for the optimal config
        cfg = advisor.suggest_config({
            "batch_size": base_args.per_device_train_batch_size,
            "learning_rate": base_args.learning_rate,
            "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp32",
        })

        # Update training args with recommended values
        optimised_args = TrainingArguments(
            output_dir=tmp,
            num_train_epochs=1,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            learning_rate=cfg.learning_rate,
            bf16=cfg.precision == "bf16" and torch.cuda.is_available(),
            gradient_checkpointing=cfg.use_gradient_checkpointing,
            logging_steps=10,
            save_steps=500,
            no_cuda=not torch.cuda.is_available(),
            use_cpu=not torch.cuda.is_available(),
            report_to=[],
        )

        callback = SysPlugTrainerCallback(advisor)

        trainer = Trainer(
            model=model,
            args=optimised_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[callback],
        )

        print("\nStarting training...")
        trainer.train()
        print("\n[DONE] SFT training complete.")


if __name__ == "__main__":
    main()
