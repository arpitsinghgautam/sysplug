"""DeepSpeed integration for SysPlug.

Provides :func:`patch_deepspeed_config` which merges SysPlug's recommended
settings into a DeepSpeed config dict consistently.

Requires ``pip install sysplug[deepspeed]``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sysplug.advisor import Advisor


def patch_deepspeed_config(
    ds_config: dict[str, Any],
    advisor: Advisor,
) -> dict[str, Any]:
    """Merge SysPlug recommendations into a DeepSpeed config dict.

    Reads the advisor's current :class:`~sysplug.config.SysPlugConfig` and
    sets the corresponding DeepSpeed fields, emitting warnings for any
    conflicts with the existing config.

    Args:
        ds_config: An existing DeepSpeed configuration dictionary.  This
            dict is **not** mutated; a new dict is returned.
        advisor: A :class:`~sysplug.advisor.Advisor` with a current config.

    Returns:
        A new DeepSpeed config dict with SysPlug settings applied.

    Raises:
        RuntimeError: If the advisor has no current config (call
            ``suggest_config`` first).

    Examples:
        >>> import sysplug
        >>> advisor = sysplug.Advisor(model="gpt2")
        >>> _ = advisor.suggest_config({"batch_size": 4})
        >>> ds = patch_deepspeed_config({}, advisor)
        >>> ds["train_micro_batch_size_per_gpu"]
        4
    """
    cfg = advisor.current_config
    if cfg is None:
        raise RuntimeError("Advisor has no current config. Call advisor.suggest_config() first.")

    patched: dict[str, Any] = dict(ds_config)

    # --- Batch sizes ---
    existing_micro = patched.get("train_micro_batch_size_per_gpu")
    if existing_micro is not None and existing_micro != cfg.batch_size:
        warnings.warn(
            f"[SysPlug] Overriding DeepSpeed train_micro_batch_size_per_gpu "
            f"{existing_micro} → {cfg.batch_size}.",
            stacklevel=2,
        )
    patched["train_micro_batch_size_per_gpu"] = cfg.batch_size

    existing_ga = patched.get("gradient_accumulation_steps")
    if existing_ga is not None and existing_ga != cfg.gradient_accumulation:
        warnings.warn(
            f"[SysPlug] Overriding DeepSpeed gradient_accumulation_steps "
            f"{existing_ga} → {cfg.gradient_accumulation}.",
            stacklevel=2,
        )
    patched["gradient_accumulation_steps"] = cfg.gradient_accumulation

    patched["train_batch_size"] = cfg.batch_size * cfg.gradient_accumulation * cfg.gpu_count

    # --- Precision ---
    if cfg.precision == "bf16":
        if "fp16" in patched and patched["fp16"].get("enabled"):
            warnings.warn(
                "[SysPlug] SysPlug recommends bf16 but fp16 is enabled in the "
                "DeepSpeed config; overriding to bf16.",
                stacklevel=2,
            )
        patched["bf16"] = {"enabled": True}
        patched.pop("fp16", None)
    elif cfg.precision == "fp16":
        patched["fp16"] = {"enabled": True}
        patched.pop("bf16", None)
    else:
        patched.pop("bf16", None)
        patched.pop("fp16", None)

    # --- ZeRO stage ---
    zero_stage_map: dict[str, int] = {
        "zero1": 1,
        "zero2": 2,
        "zero3": 3,
        "fsdp": 3,
        "none": 0,
        "dp": 0,
        "ddp": 0,
    }
    zero_stage = zero_stage_map.get(cfg.parallelism, 0)

    if zero_stage > 0:
        existing_stage = patched.get("zero_optimization", {}).get("stage")
        if existing_stage is not None and existing_stage != zero_stage:
            warnings.warn(
                f"[SysPlug] Overriding DeepSpeed ZeRO stage {existing_stage} → {zero_stage}.",
                stacklevel=2,
            )
        patched.setdefault("zero_optimization", {})["stage"] = zero_stage

    return patched
