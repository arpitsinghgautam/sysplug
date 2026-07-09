# SysPlug Usage Guide

SysPlug is a GPU-aware hyperparameter advisor for deep learning training. It
profiles your hardware, models memory and throughput analytically, and returns
a safe, optimised `SysPlugConfig` — no manual tuning required.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [The Advisor](#the-advisor)
   - [Creating an Advisor](#creating-an-advisor)
   - [suggest_config()](#suggest_config)
   - [what_if()](#what_if)
4. [SysPlugConfig](#sysplugconfig)
   - [Fields](#fields)
   - [Framework Adapters](#framework-adapters)
5. [Online Monitor](#online-monitor)
6. [Framework Integrations](#framework-integrations)
   - [Hugging Face Transformers](#hugging-face-transformers)
   - [DeepSpeed](#deepspeed)
   - [Raw PyTorch](#raw-pytorch)
   - [RLHF / PPO](#rlhf--ppo)
7. [Hardware Profiling](#hardware-profiling)
8. [Memory Model](#memory-model)
9. [Throughput Model](#throughput-model)
10. [Stability Signal](#stability-signal)
11. [CLI Reference](#cli-reference)
12. [Configuration Reference](#configuration-reference)
13. [Common Patterns](#common-patterns)
14. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Core library (no optional deps)
pip install sysplug

# With Hugging Face Transformers support
pip install "sysplug[hf]"

# With DeepSpeed support
pip install "sysplug[deepspeed]"

# Everything (development)
pip install "sysplug[hf,deepspeed,dev]"
```

---

## Quick Start

```python
import sysplug

# 1. Create an advisor for your model
advisor = sysplug.Advisor(model="llama-3-8b", training_type="sft")

# 2. Get a recommended configuration
cfg = advisor.suggest_config({
    "batch_size": 8,
    "learning_rate": 2e-5,
    "precision": "bf16",
    "optimizer": "adamw",
})

print(cfg.summary())
# SysPlug Recommended Config
# +--------------------------+----------+
# | Parameter                | Value    |
# +--------------------------+----------+
# | batch_size               | 8        |
# | gradient_accumulation    | 1        |
# | effective_batch_size     | 8        |
# | learning_rate            | 2.00e-05 |
# | precision                | bf16     |
# | optimizer                | adamw    |
# | pred. peak memory (MB)   | 18432.0  |
# | pred. throughput (samp/s)| 12.3     |
# +--------------------------+----------+

# 3. Use the config with your training framework
training_args = cfg.to_training_arguments(output_dir="./checkpoints")
```

---

## The Advisor

### Creating an Advisor

```python
import sysplug

# From a model name string (uses parameter count lookup table)
advisor = sysplug.Advisor(model="gpt2")

# From a parameter count integer
advisor = sysplug.Advisor(model=125_000_000)

# From a torch.nn.Module (counts parameters automatically)
import torch
my_model = torch.nn.TransformerEncoder(...)
advisor = sysplug.Advisor(model=my_model)

# Choose training type (affects LR scaling rule)
advisor = sysplug.Advisor(
    model="llama-3-8b",
    training_type="sft",       # "supervised", "sft", "dpo", "rlhf", "grpo"
    objective="balanced",      # "throughput", "memory", "balanced"
    verbose=True,              # print rich-formatted output
)

# Use a specific GPU or subset of GPUs
advisor = sysplug.Advisor(model="gpt2", device_ids=[0, 1])

# Inject custom constraints
from sysplug.solver import SolverConstraints
constraints = SolverConstraints(
    memory_safety_factor=0.80,  # reserve 20% VRAM headroom
    min_batch_size=2,
    max_grad_accumulation=32,
)
advisor = sysplug.Advisor(model="gpt2", constraints=constraints)

# Inject a known hardware snapshot (useful for testing / CI)
from sysplug.hardware import HardwareSnapshot, GPUSnapshot
hw = HardwareSnapshot(
    gpus=[GPUSnapshot(0, "A100", 40_960, 0, 40_960, 0, 0, (8, 0), 2039)],
    cpu_count=16, ram_total_mb=131_072,
)
advisor = sysplug.Advisor(model="gpt2", hardware=hw)
```

### suggest_config()

`suggest_config()` runs the full pipeline and returns a `SysPlugConfig`:

1. Refresh GPU hardware snapshot
2. Validate and normalise your input dict
3. Run the constrained OOM-recovery solver
4. Scale the learning rate if effective batch changed
5. Emit stability warnings for risky combinations

```python
cfg = advisor.suggest_config({
    # All keys are optional — sensible defaults are filled in
    "batch_size": 8,
    "gradient_accumulation": 1,
    "learning_rate": 2e-5,
    "precision": "bf16",         # fp32 / fp16 / bf16 / int8 / int4
    "optimizer": "adamw",        # adamw / adam / sgd / adafactor
    "parallelism": "none",       # none / dp / ddp / zero1 / zero2 / zero3 / fsdp
    "use_gradient_checkpointing": False,
    "sequence_length": 512,
})

print(f"Batch size:    {cfg.batch_size}")
print(f"Grad acc:      {cfg.gradient_accumulation}")
print(f"Effective bs:  {cfg.effective_batch_size}")
print(f"Learning rate: {cfg.learning_rate:.2e}")
print(f"Precision:     {cfg.precision}")
print(f"Peak mem (MB): {cfg.predicted_peak_memory_mb:.0f}")
print(f"Throughput:    {cfg.predicted_throughput_samples_per_sec:.1f} samples/s")

if cfg.warnings:
    for w in cfg.warnings:
        print(f"  WARNING: {w}")
```

**OOM recovery** — if the requested config doesn't fit in GPU VRAM, the solver
automatically:
1. Enables gradient checkpointing
2. Halves `batch_size` and doubles `gradient_accumulation` (preserving effective batch)
3. Downgrades precision (fp32 → fp16 → bf16 → int8 → int4) as a last resort

```python
# Request a large batch on a small GPU
cfg = advisor.suggest_config({"batch_size": 128, "precision": "fp32"})

# Solver recovers automatically:
print(cfg.notes)
# ["Enabled gradient checkpointing to reduce activation memory.",
#  "Reduced batch_size 128 → 64 (grad_acc now 2) to fit in GPU memory."]
```

### what_if()

Evaluate a proposed hyperparameter change without modifying the current config:

```python
# First get a baseline config
cfg = advisor.suggest_config({"batch_size": 4, "learning_rate": 1e-4})

# What if we doubled the batch size?
result = advisor.what_if({"batch_size": 8})

print(f"Feasible: {result.feasible}")
print(f"New batch: {result.new_config.batch_size}")
print(f"New LR:    {result.new_config.learning_rate:.2e}")  # scaled automatically

# Inspect what changed
for param, (old, new) in result.changed_params.items():
    reason = result.reason[param]
    print(f"  {param}: {old} → {new}  ({reason})")

# What if we switched to fp16 AND increased batch to 32?
result = advisor.what_if({"precision": "fp16", "batch_size": 32})
if not result.feasible:
    print("Warning: config may not fit in GPU memory")
```

---

## SysPlugConfig

### Fields

| Field | Type | Description |
|---|---|---|
| `batch_size` | `int` | Per-device micro-batch size |
| `gradient_accumulation` | `int` | Gradient accumulation steps |
| `effective_batch_size` | `int` | `batch × acc × gpu_count` |
| `learning_rate` | `float` | Recommended learning rate |
| `precision` | `str` | `fp32` / `fp16` / `bf16` / `int8` / `int4` |
| `optimizer` | `str` | `adamw` / `adam` / `sgd` / `adafactor` |
| `parallelism` | `str` | `none` / `dp` / `ddp` / `zero1`-`3` / `fsdp` |
| `use_gradient_checkpointing` | `bool` | Whether GC is advised |
| `predicted_peak_memory_mb` | `float` | Estimated peak VRAM in MiB |
| `predicted_throughput_samples_per_sec` | `float` | Estimated samples/sec |
| `warnings` | `list[str]` | Human-readable warnings |
| `notes` | `list[str]` | Informational notes from solver |

### Framework Adapters

#### `to_training_arguments()` — Hugging Face

```python
from transformers import Trainer

training_args = cfg.to_training_arguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    eval_strategy="epoch",  # `evaluation_strategy` was renamed in transformers>=4.46
    save_strategy="epoch",
    logging_steps=10,
)
trainer = Trainer(model=model, args=training_args, ...)
```

#### `to_deepspeed_config()` — DeepSpeed

```python
ds_config = cfg.to_deepspeed_config()
# Returns a dict ready to pass as deepspeed_config=...

# Or merge into an existing config
base_ds = {"scheduler": {"type": "WarmupLR", "params": {"warmup_num_steps": 100}}}
ds_config = cfg.to_deepspeed_config(base_config=base_ds)
```

#### `apply_to_optimizer()` — PyTorch

```python
import torch

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Overwrite LR with SysPlug's recommendation
cfg.apply_to_optimizer(optimizer)
print(optimizer.param_groups[0]["lr"])  # cfg.learning_rate
```

#### `to_dict()` — Serialisation

```python
import json

cfg_dict = cfg.to_dict()
with open("sysplug_config.json", "w") as f:
    json.dump(cfg_dict, f, indent=2)
```

---

## Online Monitor

The `Monitor` runs in a background thread and watches for GPU memory pressure
and training instability while your training loop runs.

```python
advisor = sysplug.Advisor(model="llama-3-8b", training_type="sft")
cfg = advisor.suggest_config({"batch_size": 8})

with advisor.monitor(
    check_interval_steps=100,       # how often to run checks
    reconfig_policy="suggest",      # "suggest" | "auto-apply" | "warn-only"
) as mon:
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)

        # Non-blocking — never slows down your training loop
        mon.record(
            step=step,
            loss=loss.item(),
            grad_norm=grad_norm,    # optional
        )

# After training, inspect what happened
events = mon.get_events()
for event in events:
    print(f"[step {event.step}] {event.event_type}: {event.message}")
```

### Reconfig policies

| Policy | Behaviour |
|---|---|
| `"suggest"` | Prints a reconfiguration suggestion to console |
| `"auto-apply"` | Automatically updates `advisor.current_config` |
| `"warn-only"` | Logs a warning, takes no action |

### Event types

```python
from sysplug.monitor import EventType

# EventType.OOM_RISK           — GPU memory > 90%
# EventType.DIVERGING_LOSS     — loss trending upward > 20% above window min
# EventType.OSCILLATING_LOSS   — normalised loss variance > threshold
# EventType.GRAD_NORM_SPIKE    — grad norm > mean + 3σ
# EventType.RECONFIG_SUGGESTED — a new config was printed
# EventType.RECONFIG_APPLIED   — config was auto-applied (auto-apply policy)

oom_events = [e for e in events if e.event_type == EventType.OOM_RISK]
```

---

## Framework Integrations

### Hugging Face Transformers

```python
from transformers import Trainer, TrainingArguments
from sysplug.integrations.huggingface import SysPlugTrainerCallback
import sysplug

# Option 1: Create callback from scratch
advisor = sysplug.Advisor(model=my_model, training_type="sft")
callback = SysPlugTrainerCallback(advisor)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    bf16=True,
    num_train_epochs=3,
)
trainer = Trainer(
    model=my_model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[callback],
)
trainer.train()
# SysPlug will:
#   - call suggest_config() at on_train_begin
#   - record loss at every step
#   - emit warnings if instability is detected at epoch end

# Option 2: Create from TrainingArguments directly
callback = SysPlugTrainerCallback.from_training_args(training_args, model=my_model)
```

### DeepSpeed

```python
from sysplug.integrations.deepspeed import patch_deepspeed_config
import sysplug
from sysplug.integrations.deepspeed import patch_deepspeed_config

advisor = sysplug.Advisor(model="llama-2-7b", training_type="sft")
cfg = advisor.suggest_config({"batch_size": 4, "parallelism": "zero3"})

# Merge SysPlug settings into your DeepSpeed config.
# Signature is patch_deepspeed_config(ds_config, advisor) — config first.
base_ds_config = {
    "scheduler": {"type": "WarmupLR"},
    "gradient_clipping": 1.0,
}
ds_config = patch_deepspeed_config(base_ds_config, advisor)
# ds_config now has batch_size, precision flags, ZeRO stage set by SysPlug
```

### Raw PyTorch

```python
from sysplug.integrations.pytorch import SysPlugContext
import torch

advisor = sysplug.Advisor(model=my_model, training_type="supervised")
cfg = advisor.suggest_config({"batch_size": 8})

optimizer = torch.optim.AdamW(my_model.parameters(), lr=cfg.learning_rate)
cfg.apply_to_optimizer(optimizer)

with SysPlugContext(advisor, check_interval_steps=50) as ctx:
    for step, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = my_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1.0)
        optimizer.step()

        # Record metrics (non-blocking)
        ctx.record(step=step, loss=loss.item(), grad_norm=grad_norm.item())
```

### RLHF / PPO

```python
from sysplug.integrations.rlhf import RLHFAdvisor, PPOConfigHelper
import sysplug

# RLHFAdvisor extends Advisor with reward-specific monitoring
advisor = RLHFAdvisor(model="llama-2-7b")
cfg = advisor.suggest_config({"batch_size": 8})

# During PPO training
for step in range(num_steps):
    # ... generate responses, compute rewards ...
    # reward_mean / reward_std / kl come from your rollout batch:
    advisor.record_reward(step, mean_reward=reward_mean, reward_std=reward_std)
    advisor.record_kl(step, kl_divergence=kl)

    # Check for reward hacking (high reward + high KL)
    if advisor.detect_reward_hacking():
        print("Warning: possible reward hacking detected")

print(advisor.reward_summary())
# {"mean_reward": 0.72, "reward_std": 0.1, "latest_kl": 0.04,
#  "reward_hacking_suspected": False}

# PPO-specific config helper. suggest(ppo_epochs=4, num_envs=1)
ppo_config = PPOConfigHelper(advisor).suggest(ppo_epochs=4, num_envs=8)
print(f"PPO rollout batch: {ppo_config.rollout_batch_size}")
print(f"PPO mini-batch:    {ppo_config.mini_batch_size}")
print(f"PPO epochs:        {ppo_config.ppo_epochs}")
```

---

## Hardware Profiling

```python
from sysplug.hardware import HardwareProfiler

profiler = HardwareProfiler()
snap = profiler.snapshot()

if snap.is_cpu_only:
    print("No CUDA GPUs found. Running in CPU-only mode.")
else:
    for gpu in snap.gpus:
        print(f"GPU {gpu.device_id}: {gpu.gpu_name}")
        print(f"  Total VRAM:  {gpu.total_memory_mb:.0f} MiB")
        print(f"  Free VRAM:   {gpu.free_memory_mb:.0f} MiB")
        print(f"  Utilisation: {gpu.gpu_utilization_pct:.0f}%")
        print(f"  Bandwidth:   {gpu.bandwidth_gbps:.0f} GB/s")

    print(f"\nTotal GPU count: {snap.gpu_count}")
    print(f"Min free memory: {snap.min_free_memory_mb():.0f} MiB")
    print(f"Avg utilisation: {snap.avg_utilization_pct():.1f}%")

# Poll continuously
import time
for _ in range(5):
    snap = profiler.poll()
    print(f"GPU 0 free: {snap.gpus[0].free_memory_mb:.0f} MiB")
    time.sleep(1)
```

---

## Memory Model

Use the `MemoryModel` directly if you want to predict memory without running
the full solver:

```python
from sysplug.memory_model import MemoryModel

mm = MemoryModel(gpu_count=1)

est = mm.predict(
    param_count=7_000_000_000,   # 7B model
    batch_size=4,
    precision="bf16",
    optimizer="adamw",
    parallelism="none",
    use_gradient_checkpointing=False,
    sequence_length=2048,
)

print(f"Central:      {est.peak_memory_mb:.0f} MiB")
print(f"Band:         [{est.lower_mb:.0f}, {est.upper_mb:.0f}] MiB (upper = OOM-safe)")
print(f"Parameters:   {est.breakdown.parameters_mb:.0f} MiB")
print(f"Gradients:    {est.breakdown.gradients_mb:.0f} MiB")
print(f"Opt. states:  {est.breakdown.optimizer_states_mb:.0f} MiB")
print(f"Activations:  {est.breakdown.activations_mb:.0f} MiB")
print(f"Overhead:     {est.breakdown.framework_overhead_mb:.0f} MiB")

# est.upper_mb is a conservative bound: if it fits your VRAM, the run fits.
# `SysPlugConfig.predicted_peak_memory_upper_mb` exposes the same for the solver.

# Attention-aware: pass the real architecture for an accurate estimate. The
# Advisor does this automatically when you give it a real nn.Module; here you
# can pass it explicitly. FlashAttention/SDPA drop the O(S^2) scores term.
est_flash = mm.predict(
    7_000_000_000, batch_size=4, sequence_length=8192,
    hidden_dim=4096, num_layers=32, num_heads=32, attn_impl="flash",
)

# Named models shortcut (uses the per-family arch table, incl. GQA)
est = mm.predict_from_name("llama-3-8b", batch_size=4, precision="bf16")

# ZeRO-3 with 8 GPUs
mm_z3 = MemoryModel(gpu_count=8)
est_z3 = mm_z3.predict(7_000_000_000, 4, "bf16", "adamw", "zero3")
print(f"ZeRO-3 per-GPU: {est_z3.peak_memory_mb:.0f} MiB")

# Calibrate against real measurements
samples = [
    {"param_count": 125_000_000, "batch_size": 4, "precision": "bf16",
     "optimizer": "adamw", "parallelism": "none", "measured_mb": 3200.0},
    {"param_count": 125_000_000, "batch_size": 8, "precision": "bf16",
     "optimizer": "adamw", "parallelism": "none", "measured_mb": 4100.0},
]
factor = mm.calibrate(samples)
print(f"Calibration factor: {factor:.3f}")  # now applied to all future predictions
```

### Supported precisions

| Precision | Bytes/param | Notes |
|---|---|---|
| `fp32` | 4 | Full precision, large memory |
| `fp16` | 2 | Mixed precision; can overflow |
| `bf16` | 2 | Mixed precision; recommended for modern GPUs |
| `int8` | 1 | Quantised inference/training |
| `int4` | 0.5 | Aggressive quantisation |

### Memory components

```
Total = params + gradients + optimizer_states + activations + overhead(500 MB)

AdamW optimizer states = 3 × fp32_params  (m + v + master weights)
SGD optimizer states   = 0
Adafactor              = 0.5 × fp32_params

ZeRO-1: shards optimizer states  ÷ gpu_count
ZeRO-2: also shards gradients    ÷ gpu_count
ZeRO-3: also shards parameters   ÷ gpu_count
FSDP:   same as ZeRO-3
```

---

## Throughput Model

```python
from sysplug.throughput_model import ThroughputModel

tm = ThroughputModel(gpu_name="A100", gpu_count=1)

est = tm.predict(
    effective_batch_size=32,
    model_size_params=125_000_000,
    precision="bf16",
    sequence_length=512,
)

print(f"Throughput:  {est.samples_per_sec:.1f} samples/sec")
print(f"Tokens/sec:  {est.tokens_per_sec:.0f}")
print(f"FLOPs/step:  {est.flops_per_step:.2e}")
print(f"Mem-bound:   {est.is_memory_bound}")
print(f"TFLOPS (att):{est.attainable_tflops:.1f}")

# Calibrate from real measurements
samples = [
    {"effective_batch_size": 8,  "model_size_params": 125_000_000,
     "precision": "bf16", "sequence_length": 512,
     "measured_samples_per_sec": 45.2},
    {"effective_batch_size": 16, "model_size_params": 125_000_000,
     "precision": "bf16", "sequence_length": 512,
     "measured_samples_per_sec": 83.1},
]
tm.calibrate_roofline(samples)
```

---

## Stability Signal

Use `StabilitySignal` standalone to watch for training instability in any loop:

```python
from sysplug.stability import StabilitySignal

signal = StabilitySignal(
    window_size=50,           # rolling window of steps
    diverge_threshold=0.20,   # 20% increase from window min = diverging
    oscillate_threshold=0.05, # normalised variance threshold
    grad_norm_sigma=3.0,      # standard deviations for spike detection
)

for step, loss in enumerate(losses):
    signal.record_loss(step, loss)
    signal.record_grad_norm(step, grad_norm)  # optional

    if step % 50 == 0:
        report = signal.check()
        print(f"Step {step}: action={report.recommended_action}")

        if report.is_diverging:
            print(f"  Loss diverging: {report.message}")
        if report.is_oscillating:
            print(f"  Loss oscillating: {report.message}")
        if report.gradient_norm_spike:
            print(f"  Grad norm spike: {report.message}")

# Recommended actions
# "ok"                  — training is stable
# "reduce_lr"           — loss is diverging
# "reduce_batch"        — loss is oscillating
# "increase_grad_clip"  — diverging + grad norm spike
```

---

## CLI Reference

```bash
# Show all commands
python -m sysplug --help

# Get a config recommendation
python -m sysplug suggest \
    --model llama-3-8b \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --precision bf16 \
    --optimizer adamw \
    --training-type sft \
    --objective balanced

# Show hardware summary
python -m sysplug hardware

# Show version
python -m sysplug version
```

**suggest** options:

| Option | Default | Description |
|---|---|---|
| `--model` | `gpt2` | Model name or param count |
| `--batch-size` | `8` | Per-device batch size |
| `--learning-rate` | `1e-4` | Initial learning rate |
| `--precision` | `bf16` | Training precision |
| `--optimizer` | `adamw` | Optimizer type |
| `--grad-acc` | `0` | Gradient accumulation steps (0 = auto) |
| `--training-type` | `supervised` | `supervised` / `sft` / `dpo` / `rlhf` / `grpo` |
| `--objective` | `balanced` | `throughput` / `memory` / `balanced` |

---

## Configuration Reference

### `SolverConstraints`

```python
from sysplug.solver import SolverConstraints

constraints = SolverConstraints(
    memory_safety_factor=0.85,  # max VRAM fraction to use (default 0.85)
    min_gpu_util=0.0,           # minimum acceptable GPU utilisation
    max_grad_accumulation=64,   # upper limit on gradient accumulation
    min_batch_size=1,           # minimum allowable per-device batch size
    max_batch_size=None,        # maximum batch size (None = no limit)
)
```

### Input config dict keys

All keys are optional; any omitted key is filled with a sensible default.

| Key | Type | Valid values | Default |
|---|---|---|---|
| `batch_size` | `int` | `>= 1` | `8` |
| `gradient_accumulation` | `int` | `>= 1` | `1` |
| `learning_rate` | `float` | `> 0` | `1e-4` |
| `precision` | `str` | `fp32/fp16/bf16/int8/int4` | `bf16` |
| `optimizer` | `str` | `adamw/adam/sgd/adafactor` | `adamw` |
| `parallelism` | `str` | `none/dp/ddp/zero1/zero2/zero3/fsdp` | `none` |
| `use_gradient_checkpointing` | `bool` | `True/False` | `False` |
| `sequence_length` | `int` | `>= 1` | `512` |

---

## Common Patterns

### Lock specific parameters

Use `locked_params` in `what_if()` or pass them as part of your config to
prevent the solver from changing particular values:

```python
# Prevent solver from changing precision
result = advisor.what_if(
    {"batch_size": 32},
    current_config=cfg,
)
# The solver won't touch precision even if it would help memory

# In suggest_config, just omit the key if you want the solver to choose it,
# or lock it via what_if locked_params:
result = advisor.what_if(
    {"batch_size": 32},
)
# result.reason["batch_size"] == "explicitly requested by user"
```

### Profile a real training run

```python
advisor = sysplug.Advisor(model=my_model, training_type="sft")
cfg = advisor.suggest_config({"batch_size": 8})

# Run 5 real training steps to calibrate the models
profile = advisor.profile_run(dataloader=train_loader, steps=5)
print(f"Measured memory:    {profile['measured_memory_mb']:.0f} MiB")
print(f"Measured throughput:{profile['measured_samples_per_sec']:.1f} samples/s")
print(f"Calibration factor: {profile.get('calibration_factor_memory', 1.0):.3f}")

# Future predictions are now calibrated to your hardware
cfg2 = advisor.suggest_config({"batch_size": 16})
```

### Multi-GPU ZeRO training

```python
import torch
advisor = sysplug.Advisor(
    model="llama-2-7b",
    training_type="sft",
    hardware="auto",   # detects all GPUs automatically
)

cfg = advisor.suggest_config({
    "batch_size": 4,
    "parallelism": "zero3",
    "precision": "bf16",
})

ds_config = cfg.to_deepspeed_config()
print(f"ZeRO stage: {ds_config['zero_optimization']['stage']}")
print(f"Per-GPU memory: {cfg.predicted_peak_memory_mb:.0f} MiB")
print(f"GPU count: {cfg.gpu_count}")
```

### Comparing training types

```python
for tt in ["supervised", "sft", "dpo", "rlhf"]:
    adv = sysplug.Advisor(model="llama-3-8b", training_type=tt, verbose=False)
    cfg = adv.suggest_config({"batch_size": 4, "learning_rate": 1e-4})
    print(f"{tt:12s}: lr={cfg.learning_rate:.2e}  batch={cfg.batch_size}")
```

### CI / testing without a GPU

```python
from sysplug.hardware import HardwareSnapshot, GPUSnapshot

# Inject a known hardware snapshot so tests are deterministic
mock_gpu = GPUSnapshot(
    device_id=0, gpu_name="A100",
    total_memory_mb=40_960, used_memory_mb=0, free_memory_mb=40_960,
    gpu_utilization_pct=0, memory_utilization_pct=0,
    compute_capability=(8, 0), bandwidth_gbps=2039,
)
hw = HardwareSnapshot(gpus=[mock_gpu], cpu_count=16, ram_total_mb=131_072)
advisor = sysplug.Advisor(model="gpt2", hardware=hw, verbose=False)
cfg = advisor.suggest_config({"batch_size": 4})
assert cfg.batch_size >= 1
```

---

## Troubleshooting

### "Could not find a feasible configuration within constraints"

The model is too large to fit even with gradient checkpointing + precision
downgrade + batch_size=1. Solutions:

- Use ZeRO parallelism across more GPUs: `parallelism="zero3"`
- Reduce sequence length
- Use a smaller model or quantised weights (int8/int4)
- Increase `SolverConstraints.memory_safety_factor` toward 1.0 (less headroom)

### Predictions are too pessimistic / optimistic

Calibrate the memory model against a real run:

```python
cfg = advisor.suggest_config({"batch_size": 4})
profile = advisor.profile_run(dataloader, steps=5)
# After calibration, predictions adjust automatically
```

### `UnicodeEncodeError` on Windows

Ensure your terminal supports UTF-8 output, or use `verbose=False` and call
`cfg.summary(verbose=False)` for plain-text output.

### `pynvml` / `nvidia-ml-py` warning

```
FutureWarning: The pynvml package is deprecated. Install nvidia-ml-py instead.
```

```bash
pip uninstall pynvml
pip install nvidia-ml-py
```

### Learning rate seems too high / too low after batch change

The LR scaling rule depends on training type and effective batch size:

- `supervised` / `sft` with batch < 256: **linear** rule (LR ∝ batch ratio)
- `supervised` / `sft` with batch ≥ 256: **sqrt** rule (more conservative)
- `rlhf` / `grpo`: always **sqrt** rule

Override the LR after `suggest_config` if you want a specific value:

```python
cfg = advisor.suggest_config({"batch_size": 32})
cfg.learning_rate = 3e-4  # manual override
training_args = cfg.to_training_arguments(output_dir="./out")
```

---

## Supported Models (name lookup)

The following model names are recognised by `Advisor(model="...")` and
`MemoryModel.predict_from_name(...)`:

| Family | Names |
|---|---|
| GPT-2 | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` |
| LLaMA-2 | `llama-2-7b`, `llama-2-13b`, `llama-2-70b` |
| LLaMA-3 | `llama-3-8b`, `llama-3-70b` |
| Mistral | `mistral-7b`, `mixtral-8x7b` |
| Falcon | `falcon-7b`, `falcon-40b` |
| OPT | `opt-1.3b`, `opt-6.7b`, `opt-30b` |
| BLOOM | `bloom-7b1` |
| T5 | `t5-small`, `t5-base`, `t5-large`, `t5-xl`, `t5-xxl` |
| BERT | `bert-base`, `bert-large` |
| RoBERTa | `roberta-base`, `roberta-large` |
| Phi | `phi-2` |
| Gemma | `gemma-2b`, `gemma-7b` |
| Qwen | `qwen-7b`, `qwen-14b`, `qwen-72b` |
| CodeLLaMA | `codellama-7b`, `codellama-13b`, `codellama-34b` |
| DeepSeek | `deepseek-7b`, `deepseek-67b` |
| StarCoder | `starcoder`, `starcoder2-7b`, `starcoder2-15b` |

For any other model, pass the integer parameter count directly:
```python
advisor = sysplug.Advisor(model=3_000_000_000)  # 3B custom model
```
