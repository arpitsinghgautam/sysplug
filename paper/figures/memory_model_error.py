"""Memory model validation figure: predicted vs measured peak VRAM.

Plots real measurements from ``results/gpu_measurements.json`` (produced by
``python -m paper.experiments.measure_gpu``). No synthetic or mock data is used.

Usage::

    python -m paper.experiments.measure_gpu          # produce the JSON first
    python paper/figures/memory_model_error.py       # then the figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="paper/data/gpu_measurements.json")
    parser.add_argument("--output", default="memory_model_error.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(
            f"{results_path} not found. Run `python -m paper.experiments.measure_gpu` first."
        )
    data = json.loads(results_path.read_text())
    rows = data["comparison"]["rows"]
    mape = data["comparison"]["memory_mape_vs_allocated_pct"]
    gpu = data["gpu_name"]

    measured = np.array([r["measured_mib_alloc"] for r in rows]) / 1024.0  # GiB
    predicted = np.array([r["pred_mib"] for r in rows]) / 1024.0
    configs = [r["config"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: predicted vs measured scatter with the y=x ideal line.
    ax = axes[0]
    uniq = sorted(set(configs))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(uniq)))
    for cfg, color in zip(uniq, colors):
        idx = [i for i, c in enumerate(configs) if c == cfg]
        ax.scatter(measured[idx], predicted[idx], color=color, label=cfg, s=40)
    lim = max(measured.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="ideal (y=x)")
    ax.set_xlabel("Measured peak VRAM (GiB)")
    ax.set_ylabel("Predicted peak VRAM (GiB)")
    ax.set_title(f"Memory Prediction vs. Measurement\n{gpu} (MAPE {mape:.1f}%)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # Right: signed percentage error per point (shows large-batch under-prediction).
    ax = axes[1]
    err = 100.0 * (predicted - measured) / measured
    labels = [f"{r['config'].replace('gpt2-', '')}\nbs{r['batch_size']}" for r in rows]
    bar_colors = ["#4CAF50" if abs(e) <= 15 else "#FF9800" for e in err]
    ax.bar(range(len(err)), err, color=bar_colors)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhline(15, linestyle="--", color="red", alpha=0.4)
    ax.axhline(-15, linestyle="--", color="red", alpha=0.4, label="±15%")
    ax.set_xticks(range(len(err)))
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    ax.set_ylabel("Prediction error (%)")
    ax.set_title(
        "Signed error per (model, batch)\n(calibrated; conservative upper covers 100%)",
        fontsize=9,
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
