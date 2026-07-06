"""Measure real training throughput and peak memory on the local GPU, and
compare against sysplug's analytic predictions.

This is the ground-truth harness behind the paper's validation numbers. It
builds real GPT-2-family transformers from ``GPT2Config`` (random init — **no
network download**), runs actual forward+backward+optimizer steps at a sweep of
batch sizes, and records measured samples/sec and peak VRAM. It then compares
those measurements to the uncalibrated model, fits calibration, and reports the
post-calibration error.

Usage::

    python -m paper.experiments.measure_gpu --steps 20 --seq 512 \
        --configs gpt2-small gpt2-medium --out results/gpu_measurements.json

Everything is deterministic given ``--seed`` except wall-clock timing.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from sysplug.memory_model import MemoryModel
from sysplug.throughput_model import ThroughputModel

# GPT-2 family configs (n_embd, n_layer, n_head) — real architectures.
_CONFIGS: Dict[str, Dict[str, int]] = {
    "gpt2-small": {"n_embd": 768, "n_layer": 12, "n_head": 12},
    "gpt2-medium": {"n_embd": 1024, "n_layer": 24, "n_head": 16},
    "gpt2-large": {"n_embd": 1280, "n_layer": 36, "n_head": 20},
}

_MiB = 1024.0 * 1024.0


@dataclass
class Measurement:
    config: str
    param_count: int
    hidden_size: int
    num_layers: int
    batch_size: int
    seq_len: int
    precision: str
    # measured
    samples_per_sec: float
    step_time_ms: float
    peak_mib_allocated: float
    peak_mib_reserved: float
    achieved_tflops: float
    ok: bool
    error: str = ""


def _build_model(cfg_name: str, vocab_size: int = 50257) -> "torch.nn.Module":
    """Build a random-init GPT-2 model of the named size (no download)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    c = _CONFIGS[cfg_name]
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=2048,
        n_embd=c["n_embd"],
        n_layer=c["n_layer"],
        n_head=c["n_head"],
    )
    return GPT2LMHeadModel(config)


def _precision_dtype(precision: str) -> Optional[torch.dtype]:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": None,
    }[precision]


def _measure_one(
    cfg_name: str,
    batch_size: int,
    seq_len: int,
    steps: int,
    precision: str,
    device: torch.device,
    seed: int,
) -> Measurement:
    """Run a real training loop and measure throughput + peak memory."""
    torch.manual_seed(seed)
    model = _build_model(cfg_name).to(device)
    model.train()
    param_count = sum(p.numel() for p in model.parameters())
    c = _CONFIGS[cfg_name]
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dtype = _precision_dtype(precision)
    vocab = model.config.vocab_size

    base = Measurement(
        config=cfg_name,
        param_count=param_count,
        hidden_size=c["n_embd"],
        num_layers=c["n_layer"],
        batch_size=batch_size,
        seq_len=seq_len,
        precision=precision,
        samples_per_sec=0.0,
        step_time_ms=0.0,
        peak_mib_allocated=0.0,
        peak_mib_reserved=0.0,
        achieved_tflops=0.0,
        ok=False,
    )

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        def one_step() -> None:
            ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
            optim.zero_grad(set_to_none=True)
            if dtype is not None:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = model(input_ids=ids, labels=ids)
                    loss = out.loss
            else:
                out = model(input_ids=ids, labels=ids)
                loss = out.loss
            loss.backward()
            optim.step()

        # Warmup (also triggers cuDNN autotune / allocator growth)
        for _ in range(3):
            one_step()
        torch.cuda.synchronize(device)

        times: List[float] = []
        for _ in range(steps):
            t0 = time.perf_counter()
            one_step()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

        step_time = statistics.median(times)
        samples_per_sec = batch_size / step_time
        # Achieved compute: 6 * P * seq * batch FLOPs per step.
        flops = 6.0 * param_count * seq_len * batch_size
        achieved_tflops = (flops / step_time) / 1e12

        base.samples_per_sec = samples_per_sec
        base.step_time_ms = step_time * 1000.0
        base.peak_mib_allocated = torch.cuda.max_memory_allocated(device) / _MiB
        base.peak_mib_reserved = torch.cuda.max_memory_reserved(device) / _MiB
        base.achieved_tflops = achieved_tflops
        base.ok = True
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        base.error = f"OOM: {str(e)[:80]}"
    except RuntimeError as e:
        base.error = ("OOM: " if "out of memory" in str(e).lower() else "") + str(e)[:120]
    finally:
        del model, optim
        torch.cuda.empty_cache()

    return base


def run(
    configs: List[str],
    batch_sizes: List[int],
    seq_len: int,
    steps: int,
    precision: str,
    seed: int,
) -> Dict:
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    measurements: List[Measurement] = []

    for cfg in configs:
        for bs in batch_sizes:
            m = _measure_one(cfg, bs, seq_len, steps, precision, device, seed)
            status = (
                f"{m.samples_per_sec:8.1f} samp/s  {m.peak_mib_reserved:8.0f} MiB  "
                f"{m.achieved_tflops:6.1f} TFLOPS"
                if m.ok
                else f"[{m.error}]"
            )
            print(f"  {cfg:>12} bs={bs:<4} seq={seq_len}: {status}")
            measurements.append(m)
            if not m.ok and "OOM" in m.error:
                break  # larger batches will also OOM

    return {
        "gpu_name": gpu_name,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seq_len": seq_len,
        "precision": precision,
        "steps": steps,
        "measurements": [asdict(m) for m in measurements],
    }


def _compare_and_calibrate(results: Dict) -> Dict:
    """Compare measurements to sysplug predictions and fit calibration."""
    gpu_name = results["gpu_name"]
    seq_len = results["seq_len"]
    precision = results["precision"]
    ok = [m for m in results["measurements"] if m["ok"]]

    mem_model = MemoryModel(gpu_count=1)
    tput = ThroughputModel(gpu_name=gpu_name, gpu_count=1)

    rows = []
    for m in ok:
        mem_est = mem_model.predict(
            param_count=m["param_count"],
            batch_size=m["batch_size"],
            precision=precision,
            optimizer="adamw",
            sequence_length=seq_len,
            hidden_dim=m["hidden_size"],
            num_layers=m["num_layers"],
        )
        tp_est = tput.predict(
            effective_batch_size=m["batch_size"],
            model_size_params=m["param_count"],
            precision=precision,
            sequence_length=seq_len,
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
        )
        rows.append({
            "config": m["config"],
            "batch_size": m["batch_size"],
            "measured_sps": m["samples_per_sec"],
            "pred_sps_uncal": tp_est.samples_per_sec,
            "measured_mib_alloc": m["peak_mib_allocated"],
            "measured_mib_reserved": m["peak_mib_reserved"],
            "pred_mib": mem_est.peak_memory_mb,
        })

    # Fit throughput calibration per config (step-time linear fit).
    calibrated = {}
    for cfg in {r["config"] for r in rows}:
        pts = [
            {"effective_batch_size": r["batch_size"], "measured_samples_per_sec": r["measured_sps"]}
            for r in rows if r["config"] == cfg
        ]
        if len(pts) >= 2:
            t = ThroughputModel(gpu_name=gpu_name, gpu_count=1)
            t.fit_empirical(pts)
            for r in rows:
                if r["config"] == cfg:
                    params = next(
                        mm["param_count"] for mm in ok
                        if mm["config"] == cfg and mm["batch_size"] == r["batch_size"]
                    )
                    r["pred_sps_cal"] = t.predict(
                        r["batch_size"], params, precision, seq_len
                    ).samples_per_sec
            calibrated[cfg] = t._empirical_coeffs  # noqa: SLF001

    def mape(pred_key: str) -> float:
        errs = [abs(r[pred_key] - r["measured_sps"]) / r["measured_sps"]
                for r in rows if pred_key in r]
        return 100.0 * statistics.mean(errs) if errs else float("nan")

    def mem_mape(meas_key: str) -> float:
        errs = [abs(r["pred_mib"] - r[meas_key]) / r[meas_key] for r in rows]
        return 100.0 * statistics.mean(errs) if errs else float("nan")

    peak_tflops = max((m["achieved_tflops"] for m in ok), default=0.0)

    summary = {
        "throughput_mape_uncalibrated_pct": mape("pred_sps_uncal"),
        "throughput_mape_calibrated_pct": mape("pred_sps_cal"),
        "memory_mape_vs_allocated_pct": mem_mape("measured_mib_alloc"),
        "memory_mape_vs_reserved_pct": mem_mape("measured_mib_reserved"),
        "peak_achieved_tflops": peak_tflops,
        "rows": rows,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=["gpt2-small"], choices=list(_CONFIGS))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gpu_measurements.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for measurement.")

    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__} / CUDA {torch.version.cuda}")
    print(f"Sweep: configs={args.configs} batches={args.batch_sizes} seq={args.seq} "
          f"precision={args.precision} steps={args.steps}\n")

    results = run(
        args.configs, args.batch_sizes, args.seq, args.steps, args.precision, args.seed
    )
    summary = _compare_and_calibrate(results)
    results["comparison"] = summary

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n=== sysplug prediction vs measurement ===")
    print(f"{'config':>12} {'bs':>4} {'meas sps':>10} {'pred(cal)':>10} "
          f"{'meas MiB':>9} {'pred MiB':>9}")
    for r in summary["rows"]:
        print(f"{r['config']:>12} {r['batch_size']:>4} {r['measured_sps']:>10.1f} "
              f"{r.get('pred_sps_cal', float('nan')):>10.1f} "
              f"{r['measured_mib_reserved']:>9.0f} {r['pred_mib']:>9.0f}")
    print(f"\nThroughput MAPE  uncalibrated: {summary['throughput_mape_uncalibrated_pct']:.1f}%  "
          f"calibrated: {summary['throughput_mape_calibrated_pct']:.1f}%")
    print(f"Memory MAPE  vs allocated: {summary['memory_mape_vs_allocated_pct']:.1f}%  "
          f"vs reserved: {summary['memory_mape_vs_reserved_pct']:.1f}%")
    print(f"Peak achieved: {summary['peak_achieved_tflops']:.1f} TFLOPS ({args.precision})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
