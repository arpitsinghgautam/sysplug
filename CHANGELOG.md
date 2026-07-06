# Changelog

All notable changes to SysPlug are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Python 3.12 / 3.13 classifiers.

### Changed
- `ThroughputModel.fit_empirical` now fits step-time vs. batch (a saturating
  throughput curve) instead of an unbounded linear samples/sec fit.
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
