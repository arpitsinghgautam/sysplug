"""Throughput validation figure: measured vs predicted samples/sec.

Plots real measurements from ``results/gpu_measurements.json`` (produced by
``python -m paper.experiments.measure_gpu``) against SysPlug's calibrated
predictions, alongside the roofline that underlies the model. No synthetic or
mock data is used.

Usage::

    python -m paper.experiments.measure_gpu            # produce the JSON first
    python paper/figures/throughput_comparison.py      # then the figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="paper/data/gpu_measurements.json")
    parser.add_argument("--output", default="throughput_comparison.png")
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
    gpu = data["gpu_name"]
    cal_mape = data["comparison"]["throughput_mape_calibrated_pct"]

    configs = sorted({r["config"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: measured vs predicted (calibrated) samples/sec vs batch, per model.
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(configs)))
    for cfg, color in zip(configs, colors):
        pts = sorted((r for r in rows if r["config"] == cfg), key=lambda r: r["batch_size"])
        bs = [r["batch_size"] for r in pts]
        meas = [r["measured_sps"] for r in pts]
        pred = [r.get("pred_sps_cal", float("nan")) for r in pts]
        ax.plot(bs, meas, marker="o", color=color, label=f"{cfg} (measured)")
        ax.plot(bs, pred, marker="x", linestyle="--", color=color,
                label=f"{cfg} (SysPlug)")
    ax.set_xlabel("Effective Batch Size")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title(f"Measured vs. Predicted Throughput\n{gpu} (calibrated MAPE {cal_mape:.1f}%)",
                 fontsize=9)
    ax.legend(fontsize=7)

    # Right: roofline that underlies the model (batch-aware arithmetic intensity).
    ax = axes[1]
    from sysplug.throughput_model import ThroughputModel, _get_gpu_spec

    m = ThroughputModel(gpu_name=gpu)
    spec = _get_gpu_spec(gpu)
    ai = np.logspace(0, 4, 200)
    compute_roof = spec.tflops_bf16 or spec.tflops_fp16
    mem_roof = m._bandwidth_gbps * ai / 1000.0  # GB/s * (FLOP/byte) -> TFLOP/s  # noqa: SLF001
    attainable = np.minimum(compute_roof, mem_roof)
    ax.loglog(ai, mem_roof, label="Memory-bandwidth roof")
    ax.loglog(ai, [compute_roof] * len(ai), label="Compute roof")
    ax.loglog(ai, attainable, "k--", label="Attainable")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title("Roofline (weights reused across batch\n$\\Rightarrow$ intensity grows with batch)",
                 fontsize=9)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
