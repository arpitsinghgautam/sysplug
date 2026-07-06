"""Benchmark: throughput prediction accuracy.

Measures MAPE of ThroughputModel.predict() vs actual measured samples/sec
from real training steps on available hardware.

Usage::

    python tests/benchmarks/bench_throughput_prediction.py --mock
    python tests/benchmarks/bench_throughput_prediction.py          # requires GPU
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass


@dataclass
class ThroughputBenchResult:
    batch_size: int
    precision: str
    predicted_sps: float
    actual_sps: float
    error_pct: float


def _run_mock_benchmark(n: int) -> list[ThroughputBenchResult]:
    from sysplug.throughput_model import ThroughputModel

    model = ThroughputModel(gpu_name="A100")
    rng = random.Random(42)
    results = []

    for _ in range(n):
        bs = rng.choice([4, 8, 16, 32])
        prec = rng.choice(["fp16", "bf16"])
        est = model.predict(bs, 125_000_000, prec)
        actual_sps = est.samples_per_sec * rng.uniform(0.6, 1.4)
        error_pct = abs(est.samples_per_sec - actual_sps) / max(actual_sps, 1e-6) * 100
        results.append(
            ThroughputBenchResult(
                batch_size=bs,
                precision=prec,
                predicted_sps=est.samples_per_sec,
                actual_sps=actual_sps,
                error_pct=error_pct,
            )
        )
    return results


def _run_real_benchmark(n: int) -> list[ThroughputBenchResult]:
    import torch

    if not torch.cuda.is_available():
        print("No GPU available. Use --mock.")
        raise SystemExit(1)

    from sysplug.throughput_model import ThroughputModel

    hidden = 512
    model_nn = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 4),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden * 4, hidden),
    ).cuda()

    tm = ThroughputModel(gpu_name="unknown")
    results = []
    rng = random.Random(0)

    for _ in range(n):
        bs = rng.choice([4, 8, 16, 32])
        prec = rng.choice(["fp16", "bf16"])
        dtype = torch.float16 if prec == "fp16" else torch.bfloat16
        m = model_nn.to(dtype)
        x = torch.randn(bs, hidden, device="cuda", dtype=dtype)

        start = time.perf_counter()
        warmup_steps = 3
        bench_steps = 10
        for i in range(warmup_steps + bench_steps):
            if i == warmup_steps:
                start = time.perf_counter()
            out = m(x)
            out.sum().backward()
        elapsed = time.perf_counter() - start
        actual_sps = (bench_steps * bs) / elapsed

        param_count = sum(p.numel() for p in m.parameters())
        est = tm.predict(bs, param_count, prec)
        error_pct = abs(est.samples_per_sec - actual_sps) / max(actual_sps, 1e-6) * 100
        results.append(ThroughputBenchResult(bs, prec, est.samples_per_sec, actual_sps, error_pct))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Throughput prediction benchmark")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    results = _run_mock_benchmark(args.samples) if args.mock else _run_real_benchmark(args.samples)

    mape = sum(r.error_pct for r in results) / max(len(results), 1)
    print(f"Throughput MAPE: {mape:.1f}%  over {len(results)} samples")
    for r in results[:5]:
        print(
            f"  bs={r.batch_size} prec={r.precision}  "
            f"pred={r.predicted_sps:.1f}  actual={r.actual_sps:.1f}  "
            f"err={r.error_pct:.1f}%"
        )


if __name__ == "__main__":
    main()
