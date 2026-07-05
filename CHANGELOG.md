# Changelog

All notable changes to SysPlug are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Paper LaTeX source with full experimental framework.
