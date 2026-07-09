# Changelog

All notable changes to SysPlug are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Real model-architecture introspection.** `resolve_model_arch()` and a
  `ModelArch` type read the true hidden size, layers, query/KV heads and
  attention implementation from a HuggingFace `config` (with a per-family arch
  table for name inputs and param-count inference as a fallback). The `Advisor`
  now threads this through to the memory and throughput models instead of
  guessing the architecture from the parameter count.
- **Conservative, OOM-safe memory bound.** `MemoryModel` now reports an
  asymmetric confidence band; the solver checks the **upper** bound for
  feasibility, so a recommended config that "fits" genuinely fits.
  `SysPlugConfig` gains `predicted_peak_memory_upper_mb`.
- **Attention-aware activation memory.** Models the full `O(S²)` attention-score
  memory for eager attention and drops it for FlashAttention/SDPA/memory-
  efficient kernels — the dominant term at long sequence / large batch.
- `paper/experiments/calibrate_memory.py` (CPU calibration of the memory
  coefficients against the committed measurements).

### Changed
- Memory activation coefficients calibrated against real measured training peaks
  on an RTX PRO 5000: the linear term (~54 vs the theoretical ~34, which
  undercounts autograd bookkeeping and attention-kernel workspace) gives ~10%
  central MAPE, and the eager `O(S²)` coefficient (~4.6) was validated on GPT-2
  across sequence lengths 256/512/1024 (fits within ~1% at S≥512). The
  conservative upper bound covers the allocator's *reserved* peak (the real OOM
  threshold) for 100% of measured configurations, including a GQA LLaMA model.

### Fixed
- **Throughput model was batch-invariant.** The roofline used
  `bytes_traffic = 2·P·B`, which cancelled batch size out of arithmetic
  intensity and out of the predicted throughput entirely (every batch returned
  the same number). Rewrote it: weights are read once per step
  (batch-independent) and activation traffic scales with batch, plus a fixed
  per-step overhead — throughput now correctly ramps then plateaus. Validated
  on real GPT-2 training (calibrated MAPE 7.3%).
- GPU spec table: RTX 3090/3080 `bf16` was `0.0` (zeroed predicted throughput
  on those cards); corrected. Added a `bf16→fp16→fp32` fallback so a stale
  entry can never zero a prediction.
- `test_requires_transformers_installed` now truly simulates the module being
  absent (patching `sys.modules`) instead of silently passing when it is
  installed.
- Corrected crash-on-copy-paste examples in `USAGE.md`
  (`patch_deepspeed_config` argument order, `PPOConfigHelper.suggest` kwargs,
  `HardwareSnapshot` methods, `eval_strategy`, `reward_summary` keys).
- **`StabilitySignal` now flags a NaN/Inf loss as hard divergence** instead of
  silently dropping it (the classic blow-up signal was previously ignored).
- **Solver no longer downgrades trainable weights to int8/int4** during OOM
  recovery (quantized inference formats, not training precisions); the ladder
  stops at bf16.
- **Hugging Face callback records metrics in `on_log`** (the real Trainer path)
  rather than `on_step_end`, where `logs` is never provided — the stability
  check was effectively a no-op under a real Trainer.
- **`MemoryModel` architecture inference corrected** (~284 hidden / 2 layers for
  a 7B model → ~3968 / 37); shared with the throughput model.
- Removed the dead `gradient_accumulation_steps` parameter and the unused
  `SolverConstraints.min_gpu_util` / `max_batch_size` fields.
- Frontend dashboard: bind to `127.0.0.1` by default and validate/bound all
  request inputs (HTTP 422 instead of 500/overflow).

### Added
- `paper/experiments/measure_gpu.py`: a real forward+backward+optimizer harness
  that validates and calibrates the memory and throughput models against a live
  GPU (with a memory-fraction cap for clean OOM on Windows/WDDM).
- Realistic transformer arch inference (`params → hidden/layers`) and optional
  `hidden_size`/`num_layers` arguments to `ThroughputModel.predict`.
- RTX PRO 5000 (Blackwell) and other modern GPUs in the spec table.
- 15 throughput regression tests (batch-dependence, saturation, spec/arch
  guards).
- Community health files: `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, issue/PR templates; plus `.gitignore` and
  `.gitattributes`.
- Python 3.12 / 3.13 classifiers, and a `py.typed` marker (PEP 561) so the
  package's type hints ship to downstream users.
- CI: Dependabot (pip + actions) and a `.pre-commit-config.yaml`. (Code
  scanning is enabled via GitHub's default setup on the public repo.)

### Changed
- `ThroughputModel.fit_empirical` now fits step-time vs. batch (a saturating
  throughput curve) instead of an unbounded linear samples/sec fit.
- CI now blocks on strict `mypy`, the full ruff ruleset, and `ruff format
  --check`, and runs a Python 3.9–3.13 matrix plus Windows/macOS spot-checks
  (previously mypy was advisory, ruff under-selected, and Linux/3.9–3.11 only).
- `publish.yml` uses OIDC trusted publishing only (removed the contradictory
  `PYPI_TOKEN`) with a tag-equals-version guard.
- Removed the orphaned fabricated experiment scripts (`run_*_eval.sh`,
  `collect_results.py`).
- Paper: replaced fabricated/placeholder result tables (LLaMA-3-8B, Mistral-7B,
  ImageNet) and the abstract's invented headline numbers with a reproducible
  single-GPU GPT-2 validation using real measured data; figures now plot real
  measurements.
- Set the real project identity (author, URLs); removed all `your-org` /
  `SysPlug Authors` placeholders.

## [0.1.0] - 2024-01-01

### Added
- Initial release of SysPlug.
- `Advisor` class: suggest_config, what_if, monitor, profile_run.
- `SysPlugConfig` with to_dict, to_training_arguments, to_deepspeed_config, apply_to_optimizer, summary.
- `MemoryModel`: analytic GPU memory prediction with ZeRO sharding and gradient checkpointing support.
- `ThroughputModel`: roofline-based throughput prediction with empirical calibration.
- `ConfigSolver`: constrained hyperparameter optimisation with OOM recovery and LR scaling.
- `StabilitySignal`: online loss divergence and oscillation detection.
- `HardwareProfiler`: pynvml-based GPU profiling with CPU fallback.
- `Monitor`: background-thread training monitor (suggest / auto-apply / warn-only policies).
- Integrations: Hugging Face Trainer callback, DeepSpeed config patching, PyTorch context manager, RLHF helpers.
- Full test suite with unit and integration tests (>85% coverage).
- CLI: `python -m sysplug --help`, `suggest`, `hardware` sub-commands.
- Examples: raw_pytorch, sft_huggingface, dpo_custom_loop, rlhf_ppo, deepspeed_example.
- Paper LaTeX source (design / white paper).

[Unreleased]: https://github.com/arpitsinghgautam/sysplug/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/arpitsinghgautam/sysplug/releases/tag/v0.1.0
