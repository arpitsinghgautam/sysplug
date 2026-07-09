"""Calibrate the memory-model coefficients against measured training peaks.

CPU-only: reads the committed ``paper/data/gpu_measurements.json`` (real peak
VRAM measured on a GPU) and fits

* ``_ACT_LINEAR_COEF`` — the linear activation coefficient — by least squares
  against measured allocated memory, using each point's real hidden/layers and
  the attention implementation the run actually used; and
* ``_CI_UPPER_FRAC`` — the conservative upper margin — as the smallest value for
  which the upper bound covers every measured point.

This is an in-sample fit on a small dataset; treat the numbers as provisional
until the GPU re-run (which varies sequence length and records the attention
implementation, letting the O(S^2) eager term be validated independently).

Usage::

    python -m paper.experiments.calibrate_memory
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from sysplug.memory_model import _ACT_LINEAR_COEF, MemoryModel

# GPT-2 (GPT2LMHeadModel) defaults to memory-efficient SDPA attention in current
# transformers, so the measured peaks do NOT include the O(S^2) scores term.
_MEASURED_ATTN_IMPL = "sdpa"


def calibrate(results_path: str = "paper/data/gpu_measurements.json") -> dict[str, float]:
    data = json.loads(Path(results_path).read_text())
    ok = [m for m in data["measurements"] if m["ok"]]
    prec = data["precision"]
    seq = data["seq_len"]
    mm = MemoryModel(gpu_count=1)

    # Least-squares fit of the linear activation coefficient: for each point,
    # measured = fixed + (coef/base) * activation(base). Solve for coef.
    num = den = 0.0
    fixed_and_act = []
    for m in ok:
        est = mm.predict(
            param_count=m["param_count"],
            batch_size=m["batch_size"],
            precision=prec,
            optimizer="adamw",
            sequence_length=seq,
            hidden_dim=m["hidden_size"],
            num_layers=m["num_layers"],
            attn_impl=_MEASURED_ATTN_IMPL,
        )
        fixed = est.breakdown.total_mb - est.breakdown.activations_mb
        act_per_unit = est.breakdown.activations_mb / _ACT_LINEAR_COEF
        residual = m["peak_mib_allocated"] - fixed
        num += act_per_unit * residual
        den += act_per_unit * act_per_unit
        fixed_and_act.append((fixed, act_per_unit, m["peak_mib_allocated"]))

    coef = num / den
    errs, ratios = [], []
    for fixed, act_per_unit, measured in fixed_and_act:
        pred = fixed + act_per_unit * coef
        errs.append(abs(pred - measured) / measured)
        ratios.append(measured / pred)

    return {
        "act_linear_coef": coef,
        "central_mape_pct": 100 * statistics.mean(errs),
        "min_ci_upper_frac": max(ratios) - 1.0,
        "n_points": len(ok),
    }


def main() -> None:
    r = calibrate()
    print(f"points:               {r['n_points']}")
    print(f"_ACT_LINEAR_COEF fit: {r['act_linear_coef']:.1f}")
    print(f"central MAPE:         {r['central_mape_pct']:.1f}%")
    print(f"min _CI_UPPER_FRAC:   {r['min_ci_upper_frac']:.3f} (for 100% coverage)")


if __name__ == "__main__":
    main()
