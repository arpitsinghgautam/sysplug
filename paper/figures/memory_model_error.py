"""Generate memory model prediction error plot for the paper.

Usage::

    python paper/figures/memory_model_error.py --mock
    python paper/figures/memory_model_error.py  # requires GPU
"""

from __future__ import annotations

import argparse
import random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", default="memory_model_error.pdf")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    from sysplug.memory_model import MemoryModel

    rng = random.Random(42)
    model = MemoryModel()
    precisions = ["fp32", "fp16", "bf16"]
    param_counts = [125_000_000, 345_000_000, 1_300_000_000, 7_000_000_000]
    batch_sizes = [1, 2, 4, 8, 16]

    results = {"fp32": [], "fp16": [], "bf16": []}

    for prec in precisions:
        for params in param_counts:
            for bs in batch_sizes:
                pred = model.predict(params, bs, prec, "adamw").peak_memory_mb
                if args.mock:
                    actual = pred * rng.uniform(0.75, 1.25)
                else:
                    actual = pred  # placeholder

                error = abs(pred - actual) / max(actual, 1.0) * 100
                results[prec].append(error)

    fig, ax = plt.subplots(figsize=(6, 4))

    positions = [1, 2, 3]
    data = [results[p] for p in precisions]
    bp = ax.boxplot(data, positions=positions, labels=precisions, patch_artist=True)

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Precision")
    ax.set_ylabel("Absolute Percentage Error (%)")
    ax.set_title("SysPlug Memory Model Prediction Error")
    ax.axhline(y=15, linestyle="--", color="red", alpha=0.5, label="15% threshold")
    ax.legend()
    ax.set_ylim(0, 50)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
