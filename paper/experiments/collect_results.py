"""Collect experiment results from CSVs and generate LaTeX tables."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def load_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_mape(rows: list[dict[str, str]], pred_col: str, actual_col: str) -> float:
    if not rows:
        return float("nan")
    errors = [
        abs(float(r[pred_col]) - float(r[actual_col])) / max(float(r[actual_col]), 1.0) * 100
        for r in rows
    ]
    return sum(errors) / len(errors)


def main() -> None:
    results_dir = Path("results")

    # Memory benchmark
    mem_rows = load_csv(str(results_dir / "prediction" / "memory_bench.csv"))
    mape = compute_mape(mem_rows, "predicted_mb", "actual_mb")
    print(f"Memory model MAPE: {mape:.1f}%  ({len(mem_rows)} samples)")

    # Generate LaTeX snippet
    print()
    print("% LaTeX table snippet for paper:")
    print(r"\begin{tabular}{lr}")
    print(r"\toprule")
    print(r"Metric & Value \\ \midrule")
    print(f"Memory MAPE (uncalibrated) & {mape:.1f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
