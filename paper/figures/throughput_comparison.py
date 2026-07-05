"""Generate throughput comparison figure for the paper."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", default="throughput_comparison.pdf")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    from sysplug.throughput_model import ThroughputModel

    gpu_names = ["A100", "V100", "T4", "RTX 4090"]
    batch_sizes = [4, 8, 16, 32, 64]
    model_params = 7_000_000_000  # 7B

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: tokens/sec vs batch size by GPU
    ax = axes[0]
    for gpu in gpu_names:
        model = ThroughputModel(gpu_name=gpu)
        tps = [
            model.predict(bs, model_params, "bf16").tokens_per_sec / 1000
            for bs in batch_sizes
        ]
        ax.plot(batch_sizes, tps, marker="o", label=gpu)
    ax.set_xlabel("Effective Batch Size")
    ax.set_ylabel("Throughput (K tokens/s)")
    ax.set_title("Predicted Throughput (7B model, BF16)")
    ax.legend()

    # Right: roofline diagram (compute vs memory bound)
    ax = axes[1]
    model_a100 = ThroughputModel(gpu_name="A100")
    arithmetic_intensities = np.logspace(-1, 3, 100)
    compute_roof = model_a100._spec.tflops_bf16  # TFLOPS
    mem_roof = [model_a100._bandwidth_gbps * ai / 1000 for ai in arithmetic_intensities]
    compute_line = [compute_roof] * len(arithmetic_intensities)
    attainable = [min(c, m) for c, m in zip(compute_line, mem_roof)]

    ax.loglog(arithmetic_intensities, mem_roof, label="Memory bandwidth roof")
    ax.loglog(arithmetic_intensities, compute_line, label="Compute roof (A100 BF16)")
    ax.loglog(arithmetic_intensities, attainable, "k--", label="Attainable")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title("Roofline Model (A100)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
