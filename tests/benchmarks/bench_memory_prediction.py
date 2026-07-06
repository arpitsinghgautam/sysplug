"""Benchmark: memory prediction accuracy.

Measures Mean Absolute Percentage Error (MAPE) of MemoryModel.predict()
against actual torch.cuda.max_memory_allocated() after a real forward+backward.

Usage::

    python tests/benchmarks/bench_memory_prediction.py          # requires GPU
    python tests/benchmarks/bench_memory_prediction.py --mock   # synthetic data
    python tests/benchmarks/bench_memory_prediction.py --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass


@dataclass
class BenchResult:
    model_size: int
    batch_size: int
    precision: str
    predicted_mb: float
    actual_mb: float
    error_pct: float


def _run_real_benchmark(samples: list[dict]) -> list[BenchResult]:
    """Run benchmark against actual GPU memory (requires CUDA)."""
    try:
        import torch
    except ImportError:
        print("torch not available. Use --mock flag.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("No CUDA GPU available. Use --mock flag.")
        sys.exit(1)

    from sysplug.memory_model import MemoryModel

    results: list[BenchResult] = []
    model_inst = MemoryModel()

    for sample in samples:
        param_count = sample["param_count"]
        batch_size = sample["batch_size"]
        precision = sample["precision"]

        # Build a tiny model with approximately the right param count
        hidden = max(16, int((param_count / 4) ** 0.5))
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.Linear(hidden, hidden),
        ).cuda()

        if precision == "fp16":
            model = model.half()
        elif precision == "bf16":
            model = model.to(torch.bfloat16)

        x = torch.randn(batch_size, hidden, device="cuda")
        if precision == "fp16":
            x = x.half()
        elif precision == "bf16":
            x = x.to(torch.bfloat16)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        out = model(x)
        loss = out.sum()
        loss.backward()

        torch.cuda.synchronize()
        actual_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        pred = model_inst.predict(
            param_count=sum(p.numel() for p in model.parameters()),
            batch_size=batch_size,
            precision=precision,
            optimizer="adamw",
        )
        predicted_mb = pred.peak_memory_mb
        error_pct = abs(predicted_mb - actual_mb) / max(actual_mb, 1.0) * 100.0

        results.append(
            BenchResult(
                model_size=param_count,
                batch_size=batch_size,
                precision=precision,
                predicted_mb=predicted_mb,
                actual_mb=actual_mb,
                error_pct=error_pct,
            )
        )

    return results


def _run_mock_benchmark(samples: list[dict]) -> list[BenchResult]:
    """Run benchmark with synthetic 'actual' values (no GPU required)."""
    from sysplug.memory_model import MemoryModel

    model = MemoryModel()
    results: list[BenchResult] = []
    rng = random.Random(42)

    for sample in samples:
        pred = model.predict(
            param_count=sample["param_count"],
            batch_size=sample["batch_size"],
            precision=sample["precision"],
            optimizer="adamw",
        )
        predicted_mb = pred.peak_memory_mb
        # Simulate actual = prediction × (0.7 + 0.6 * rng.random())
        actual_mb = predicted_mb * (0.7 + 0.6 * rng.random())
        error_pct = abs(predicted_mb - actual_mb) / max(actual_mb, 1.0) * 100.0

        results.append(
            BenchResult(
                model_size=sample["param_count"],
                batch_size=sample["batch_size"],
                precision=sample["precision"],
                predicted_mb=predicted_mb,
                actual_mb=actual_mb,
                error_pct=error_pct,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory prediction benchmark")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data (no GPU needed)")
    parser.add_argument("--output", default="", help="Output CSV file path")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples")
    args = parser.parse_args()

    rng = random.Random(0)
    precisions = ["fp32", "fp16", "bf16"]
    param_sizes = [
        125_000_000,  # GPT-2
        345_000_000,  # GPT-2 medium
        1_300_000_000,  # OPT-1.3B
        7_000_000_000,  # LLaMA-7B
    ]
    batch_sizes = [1, 2, 4, 8, 16]

    samples = [
        {
            "param_count": rng.choice(param_sizes),
            "batch_size": rng.choice(batch_sizes),
            "precision": rng.choice(precisions),
        }
        for _ in range(args.samples)
    ]

    print(f"Running {args.samples} benchmark samples ({'mock' if args.mock else 'real GPU'})...")
    start = time.perf_counter()

    results = _run_mock_benchmark(samples) if args.mock else _run_real_benchmark(samples)

    elapsed = time.perf_counter() - start
    mape = sum(r.error_pct for r in results) / len(results) if results else 0.0
    print(f"MAPE: {mape:.1f}%  |  {len(results)} samples  |  {elapsed:.1f}s")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model_size",
                    "batch_size",
                    "precision",
                    "predicted_mb",
                    "actual_mb",
                    "error_pct",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "model_size": r.model_size,
                        "batch_size": r.batch_size,
                        "precision": r.precision,
                        "predicted_mb": f"{r.predicted_mb:.2f}",
                        "actual_mb": f"{r.actual_mb:.2f}",
                        "error_pct": f"{r.error_pct:.2f}",
                    }
                )
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
