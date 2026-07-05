"""SysPlug Visual Dashboard — FastAPI backend.

Run:
    pip install fastapi uvicorn
    python frontend/server.py

Then open  http://localhost:8000
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="SysPlug Visual Dashboard", version="1.0.0")

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    model_name: str = "gpt2"
    param_count: int = 0          # if > 0, overrides model_name lookup
    batch_size: int = 8
    gradient_accumulation: int = 1
    sequence_length: int = 512
    precision: str = "bf16"
    optimizer: str = "adamw"
    parallelism: str = "none"
    use_gradient_checkpointing: bool = False
    gpu_count: int = 1
    gpu_memory_mb: float = 40_960.0   # total VRAM of one GPU (MiB)


class InferenceRequest(BaseModel):
    model_name: str = "llama-3-8b"
    param_count: int = 0
    batch_size: int = 1           # concurrent requests
    sequence_length: int = 2048   # max context length
    precision: str = "bf16"
    gpu_count: int = 1
    gpu_memory_mb: float = 40_960.0
    kv_cache: bool = True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(STATIC / "index.html")


@app.get("/hardware")
def get_hardware():
    """Return live GPU snapshot, or CPU-only stub."""
    try:
        from sysplug.hardware import HardwareProfiler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            profiler = HardwareProfiler(verbose=False)
            snap = profiler.snapshot()

        if snap.gpus:
            return {
                "is_cpu_only": False,
                "gpus": [
                    {
                        "device_id": g.device_id,
                        "name": g.gpu_name,
                        "total_mb": g.total_memory_mb,
                        "free_mb": g.free_memory_mb,
                        "used_mb": g.used_memory_mb,
                        "util_pct": g.gpu_utilization_pct,
                        "compute_cap": f"{g.compute_capability[0]}.{g.compute_capability[1]}",
                        "bandwidth_gbps": g.bandwidth_gbps,
                    }
                    for g in snap.gpus
                ],
            }
    except Exception:
        pass

    return {"is_cpu_only": True, "gpus": []}


@app.post("/predict")
def predict(req: PredictRequest):
    """Run analytic memory + throughput prediction and return full breakdown."""
    from sysplug.memory_model import MemoryModel, _params_from_name
    from sysplug.throughput_model import ThroughputModel
    from sysplug.utils.scaling_rules import recommended_lr_rule

    # ── Resolve parameter count ──────────────────────────────────────────
    if req.param_count > 0:
        params = req.param_count
        model_label = f"{params / 1e9:.2f}B params"
    else:
        try:
            params = _params_from_name(req.model_name)
            model_label = req.model_name
        except ValueError:
            params = 125_000_000
            model_label = "gpt2 (fallback)"

    # ── Memory prediction ────────────────────────────────────────────────
    mm = MemoryModel(gpu_count=req.gpu_count)
    mem_est = mm.predict(
        param_count=params,
        batch_size=req.batch_size,
        precision=req.precision,
        optimizer=req.optimizer,
        parallelism=req.parallelism,
        use_gradient_checkpointing=req.use_gradient_checkpointing,
        sequence_length=req.sequence_length,
    )

    # ── Throughput prediction ────────────────────────────────────────────
    eff_batch = req.batch_size * req.gradient_accumulation * req.gpu_count
    gpu_name = _detect_gpu_name()
    tm = ThroughputModel(gpu_name=gpu_name, gpu_count=req.gpu_count)
    tput = tm.predict(
        effective_batch_size=eff_batch,
        model_size_params=params,
        precision=req.precision,
        sequence_length=req.sequence_length,
    )

    # ── Budget check ─────────────────────────────────────────────────────
    budget_mb = req.gpu_memory_mb * 0.85
    mem_pct = (mem_est.peak_memory_mb / req.gpu_memory_mb) * 100
    fits = mem_est.peak_memory_mb <= budget_mb
    oom_pct = (mem_est.peak_memory_mb / budget_mb) * 100   # % of *budget*

    # ── Warnings ─────────────────────────────────────────────────────────
    warns: list[str] = []
    if not fits:
        over = mem_est.peak_memory_mb - budget_mb
        warns.append(f"OOM risk: predicted {mem_est.peak_memory_mb:.0f} MiB "
                     f"exceeds {budget_mb:.0f} MiB budget by {over:.0f} MiB.")
    if req.precision == "fp16" and req.batch_size <= 4:
        warns.append("Small batch with fp16 can cause numerical instability.")
    if req.gradient_accumulation > 32:
        warns.append(f"gradient_accumulation={req.gradient_accumulation} is very high (>32).")
    if params > 10_000_000_000 and req.parallelism == "none":
        warns.append("Model >10B params — consider ZeRO-2/3 or FSDP parallelism.")

    # ── LR rule hint ─────────────────────────────────────────────────────
    lr_rule = recommended_lr_rule("supervised", eff_batch)

    bd = mem_est.breakdown
    total_bd = bd.total_mb

    return {
        "model_label": model_label,
        "params": params,
        "params_b": params / 1e9,

        # Memory
        "peak_memory_mb": mem_est.peak_memory_mb,
        "lower_mb": mem_est.lower_mb,
        "upper_mb": mem_est.upper_mb,
        "budget_mb": budget_mb,
        "mem_pct": mem_pct,          # % of total VRAM
        "oom_pct": oom_pct,          # % of budget
        "fits": fits,

        # Breakdown (MiB + fraction of total)
        "breakdown": {
            "parameters": {"mb": bd.parameters_mb,
                           "pct": 100 * bd.parameters_mb / max(total_bd, 1)},
            "gradients":  {"mb": bd.gradients_mb,
                           "pct": 100 * bd.gradients_mb / max(total_bd, 1)},
            "optimizer":  {"mb": bd.optimizer_states_mb,
                           "pct": 100 * bd.optimizer_states_mb / max(total_bd, 1)},
            "activations":{"mb": bd.activations_mb,
                           "pct": 100 * bd.activations_mb / max(total_bd, 1)},
            "overhead":   {"mb": bd.framework_overhead_mb,
                           "pct": 100 * bd.framework_overhead_mb / max(total_bd, 1)},
        },

        # Throughput
        "samples_per_sec": tput.samples_per_sec,
        "tokens_per_sec": tput.tokens_per_sec,
        "is_memory_bound": tput.is_memory_bound,
        "attainable_tflops": tput.attainable_tflops,

        # Batch
        "effective_batch_size": eff_batch,
        "lr_rule": lr_rule,

        # Misc
        "warnings": warns,
        "gpu_name": gpu_name,
    }


@app.post("/predict-inference")
def predict_inference(req: InferenceRequest):
    """Inference memory + throughput prediction."""
    from sysplug.memory_model import _params_from_name
    from sysplug.throughput_model import ThroughputModel

    # Resolve param count
    if req.param_count > 0:
        params = req.param_count
        model_label = f"{params / 1e9:.2f}B params"
    else:
        try:
            params = _params_from_name(req.model_name)
            model_label = req.model_name
        except ValueError:
            params = 8_000_000_000
            model_label = "llama-3-8b (fallback)"

    bpe = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}.get(req.precision, 2)

    # Parameter weights
    params_mb = params * bpe / 1_048_576

    # Architecture estimate for KV cache
    hidden, layers = _estimate_arch(params)

    # KV cache: 2 (K+V) * layers * seq * hidden * batch * bytes
    if req.kv_cache:
        kv_cache_mb = (2 * layers * req.sequence_length * hidden
                       * req.batch_size * bpe) / 1_048_576
    else:
        kv_cache_mb = 0.0

    # Activations (forward only — one layer at a time, much smaller than training)
    activations_mb = (req.batch_size * req.sequence_length * hidden * 2 * bpe) / 1_048_576

    overhead_mb = 500.0
    peak_mb = params_mb + kv_cache_mb + activations_mb + overhead_mb
    total_bd = max(peak_mb, 1)

    budget_mb = req.gpu_memory_mb * 0.85
    mem_pct = (peak_mb / req.gpu_memory_mb) * 100
    fits = peak_mb <= budget_mb

    # Warnings
    warns: list[str] = []
    if not fits:
        over = peak_mb - budget_mb
        warns.append(
            f"OOM risk: predicted {peak_mb:.0f} MiB exceeds "
            f"{budget_mb:.0f} MiB budget by {over:.0f} MiB."
        )
    if req.precision in ("fp16", "fp32") and params > 7_000_000_000:
        warns.append(
            f"For {params/1e9:.0f}B model consider int8/int4 "
            "quantization to reduce VRAM."
        )
    if req.kv_cache and kv_cache_mb > params_mb:
        warns.append(
            f"KV cache ({kv_cache_mb:.0f} MiB) exceeds model weights "
            f"({params_mb:.0f} MiB) — reduce batch or sequence length."
        )

    # Throughput
    gpu_name = _detect_gpu_name()
    tm = ThroughputModel(gpu_name=gpu_name, gpu_count=req.gpu_count)
    tput = tm.predict(
        effective_batch_size=req.batch_size,
        model_size_params=params,
        precision=req.precision,
        sequence_length=req.sequence_length,
    )

    return {
        "model_label": model_label,
        "params": params,
        "params_b": params / 1e9,

        "peak_memory_mb": peak_mb,
        "lower_mb": peak_mb * 0.85,
        "upper_mb": peak_mb * 1.15,
        "budget_mb": budget_mb,
        "mem_pct": mem_pct,
        "oom_pct": (peak_mb / budget_mb) * 100,
        "fits": fits,

        "breakdown": {
            "parameters":  {"mb": params_mb,     "pct": 100 * params_mb / total_bd},
            "kv_cache":    {"mb": kv_cache_mb,   "pct": 100 * kv_cache_mb / total_bd},
            "activations": {"mb": activations_mb,"pct": 100 * activations_mb / total_bd},
            "overhead":    {"mb": overhead_mb,   "pct": 100 * overhead_mb / total_bd},
        },

        "tokens_per_sec":    tput.tokens_per_sec,
        "samples_per_sec":   tput.samples_per_sec,
        "is_memory_bound":   tput.is_memory_bound,
        "attainable_tflops": tput.attainable_tflops,

        "warnings": warns,
        "gpu_name": gpu_name,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_arch(param_count: int) -> tuple[int, int]:
    """Estimate (hidden_size, num_layers) from parameter count."""
    if param_count < 200_000_000:
        return 768, 12
    if param_count < 1_000_000_000:
        return 1024, 24
    if param_count < 4_000_000_000:
        return 2048, 24
    if param_count < 10_000_000_000:
        return 4096, 32
    if param_count < 20_000_000_000:
        return 5120, 40
    return 8192, 80


def _detect_gpu_name() -> str:
    """Return the first detected GPU name, or 'A100' as a sensible default."""
    try:
        from sysplug.hardware import HardwareProfiler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snap = HardwareProfiler(verbose=False).snapshot()
        if snap.gpus:
            return snap.gpus[0].gpu_name
    except Exception:
        pass
    return "A100"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  SysPlug Visual Dashboard")
    print("  -----------------------------------------")
    print("  Open  http://localhost:8000  in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="warning")
