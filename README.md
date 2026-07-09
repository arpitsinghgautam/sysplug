# SysPlug

[![CI](https://github.com/arpitsinghgautam/sysplug/actions/workflows/tests.yml/badge.svg)](https://github.com/arpitsinghgautam/sysplug/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

<!-- The PyPI and Codecov badges are intentionally omitted until the package is
     published to PyPI and the repository is connected to codecov.io. -->

![SysPlug architecture](https://raw.githubusercontent.com/arpitsinghgautam/sysplug/main/docs/architecture.svg)

**GPU-aware hyperparameter advisor for any deep learning training loop.**

SysPlug analyses your GPU hardware, estimates memory and throughput requirements, and recommends the optimal batch size, learning rate, precision, gradient accumulation, and parallelism strategy for your training run, before you waste hours hitting OOM or under-utilizing hardware.

Unlike a param-count guesser, SysPlug **reads the real model architecture** (hidden size, layers, attention heads including GQA, and the attention implementation) straight from a HuggingFace `config`, models the `O(S²)` attention-score memory that dominates long-context training, and reports a **conservative, OOM-safe** bound: if it says a config fits, it fits.

## Quick Start

```bash
pip install sysplug
```

```python
import sysplug

advisor = sysplug.Advisor(model=model, training_type="sft")

# Get an optimised configuration
cfg = advisor.suggest_config({"batch_size": 8, "learning_rate": 2e-5})

# What if I try a larger batch?
result = advisor.what_if({"batch_size": 32})
print(result.changed_params)   # shows what changed and why

# Monitor training in real time
with advisor.monitor(check_interval_steps=100) as mon:
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)
        mon.record(step=step, loss=loss.item())
```

## Installation

```bash
# Core
pip install sysplug

# With Hugging Face Trainer integration
pip install sysplug[hf]

# With DeepSpeed integration
pip install sysplug[deepspeed]

# Development
pip install sysplug[dev]
```

## API Reference

### `Advisor`

The main entry point.

```python
advisor = sysplug.Advisor(
    model,                          # nn.Module, string ("llama-3-8b"), or int
    hardware="auto",                # "auto" or HardwareSnapshot
    training_type="supervised",     # "sft"|"rlhf"|"dpo"|"grpo"|"supervised"
    objective="balanced",           # "throughput"|"memory"|"balanced"
    verbose=True,
    device_ids=None,                # list of GPU IDs; None = all
)
```

#### `suggest_config(base_config: dict) -> SysPlugConfig`

Returns the best safe configuration for the given starting point. Runs hardware profiling → memory estimation → constrained optimisation.

#### `what_if(change: dict, current_config=None) -> WhatIfResult`

Evaluates a proposed change, locking the specified parameters and re-solving the rest. Returns `WhatIfResult` with `new_config`, `changed_params`, `reason`, and `feasible`.

#### `monitor(check_interval_steps=50, reconfig_policy="suggest") -> Monitor`

Returns a context manager that runs GPU polling and stability checks in a background thread.

### `SysPlugConfig`

The output type of `suggest_config`. Key methods:

| Method | Description |
|--------|-------------|
| `to_dict()` | Serialise to a plain dict |
| `to_training_arguments(**kwargs)` | Create `transformers.TrainingArguments` |
| `to_deepspeed_config(base_config)` | Merge into a DeepSpeed config dict |
| `apply_to_optimizer(optimizer)` | Set LR on an existing optimizer |
| `summary()` | Rich-formatted table |

### Framework Integrations

#### Hugging Face Trainer

```python
from sysplug.integrations.huggingface import SysPlugTrainerCallback

trainer = Trainer(
    ...,
    callbacks=[SysPlugTrainerCallback(advisor)],
)
```

#### DeepSpeed

```python
from sysplug.integrations.deepspeed import patch_deepspeed_config

ds_config = patch_deepspeed_config(my_ds_config, advisor)
```

#### RLHF / PPO

```python
from sysplug.integrations.rlhf import RLHFAdvisor, PPOConfigHelper

advisor = RLHFAdvisor(model=policy, training_type="rlhf")
advisor.record_reward(step, mean_reward=0.7, reward_std=0.1)
advisor.record_kl(step, kl_divergence=0.3)
if advisor.detect_reward_hacking():
    print("Reward hacking suspected!")

ppo_cfg = PPOConfigHelper(advisor).suggest(ppo_epochs=4)
```

### CLI

```bash
python -m sysplug suggest --model llama-3-8b --batch-size 8 --precision bf16
python -m sysplug hardware
python -m sysplug version
```

## How SysPlug Compares

| Feature | SysPlug | W&B Sweeps | Ray Tune | Manual tuning |
|---------|---------|------------|----------|---------------|
| No training runs required | ✅ | ❌ | ❌ | ❌ |
| Memory-safe suggestions | ✅ | ❌ | ⚠ | ❌ |
| LR scaling rules | ✅ | ❌ | ❌ | Manual |
| Online instability detection | ✅ | ❌ | ❌ | ❌ |
| ZeRO / FSDP awareness | ✅ | ❌ | ❌ | Manual |
| HF Trainer integration | ✅ | ✅ | ✅ | N/A |
| What-if analysis | ✅ | ❌ | ❌ | ❌ |
| No GPU required for suggestions | ✅ | ❌ | ❌ | N/A |
| Pip installable | ✅ | ✅ | ✅ | N/A |

## How It Works

SysPlug combines three components:

1. **MemoryModel**: analytic peak-VRAM estimator that **introspects the real model** (reads hidden size, layers, query/KV heads and the attention implementation from a HuggingFace `config` instead of guessing from the parameter count), models **full O(S²) attention-score memory** for eager attention and drops it for FlashAttention/SDPA, covers optimizer states and ZeRO-0..3 / FSDP sharding, and reports a **conservative upper bound** the solver treats as OOM-safe ("if it says it fits, it fits").

2. **ThroughputModel**: roofline-based throughput predictor using GPU hardware specs (peak FLOP/s, memory bandwidth), where per-step compute is 6×params×seq×batch, weight traffic is read once per step, and activation traffic scales with batch (so throughput ramps then plateaus).

3. **ConfigSolver**: constrained solver that adjusts the configuration (batch size, gradient accumulation, precision, LR) subject to GPU memory constraints and applies literature-backed LR scaling rules (Goyal et al. 2017, Krizhevsky 2014).

The `Monitor` runs these checks in a background thread during training, detecting loss divergence and OOM risk without blocking the training loop.

## How accurate is it?

Validated against real training on an NVIDIA RTX PRO 5000 (24 GB):

- **Memory: 10.1% mean error**, and the conservative upper bound covered the true out-of-memory threshold (allocator reserved peak) for **100%** of tested configurations — GPT-2 125M–775M and a grouped-query-attention LLaMA model.
- **Throughput: 7.3% mean error** after a short one-time calibration.
- The `O(S²)` eager-attention memory term was validated independently across sequence lengths 256/512/1024.

Reproduce it yourself with `python -m paper.experiments.measure_gpu`. **Scope:** validation so far is single-GPU on GPT-2 and LLaMA-family models; broader coverage (more architectures, longer contexts, multi-GPU) is in progress. Full method and numbers are in [the paper](https://github.com/arpitsinghgautam/sysplug/blob/main/paper/paper.tex).

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev deps: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/`
5. Lint: `ruff check sysplug/`
6. Open a pull request.

Please make sure all tests pass and coverage stays above 85%.

## Citation

If you use SysPlug in research, please cite:

```bibtex
@software{sysplug2026,
  title   = {{SysPlug}: GPU-aware Hyperparameter Advisor for Deep Learning Training},
  author  = {Gautam, Arpit Singh},
  year    = {2026},
  url     = {https://github.com/arpitsinghgautam/sysplug},
  version = {0.1.0},
}
```

## License

MIT License. See [LICENSE](https://github.com/arpitsinghgautam/sysplug/blob/main/LICENSE) for details.
