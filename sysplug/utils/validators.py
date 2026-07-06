"""Input validation utilities for SysPlug config dicts.

Provides clear, user-friendly error messages for invalid configuration values
before they propagate into models or solvers.
"""

from __future__ import annotations

from typing import Any

# Allowed values for categorical fields
_VALID_PRECISION = {"fp32", "fp16", "bf16", "int8", "int4"}
_VALID_OPTIMIZER = {"adamw", "adam", "sgd", "adafactor"}
_VALID_PARALLELISM = {"none", "dp", "ddp", "fsdp", "zero1", "zero2", "zero3"}
_VALID_TRAINING_TYPES = {"supervised", "sft", "dpo", "rlhf", "grpo"}
_VALID_OBJECTIVES = {"throughput", "memory", "balanced"}


def _check_positive_int(value: Any, name: str) -> int:
    """Validate that *value* is a positive integer.

    Args:
        value: The value to check.
        name: The field name used in error messages.

    Returns:
        The value cast to ``int``.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    try:
        int_val = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if int_val <= 0:
        raise ValueError(f"{name} must be a positive integer, got {int_val}")
    return int_val


def _check_positive_float(value: Any, name: str) -> float:
    """Validate that *value* is a positive float.

    Args:
        value: The value to check.
        name: The field name used in error messages.

    Returns:
        The value cast to ``float``.

    Raises:
        ValueError: If the value is not a positive float.
    """
    try:
        float_val = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive float, got {value!r}") from None
    if float_val <= 0:
        raise ValueError(f"{name} must be a positive float, got {float_val}")
    return float_val


def _check_in_set(value: Any, name: str, valid: set[str]) -> str:
    """Validate that *value* is one of the allowed strings.

    Args:
        value: The value to check.
        name: The field name used in error messages.
        valid: Set of allowed string values.

    Returns:
        The value as a lowercased string.

    Raises:
        ValueError: If the value is not in the allowed set.
    """
    str_val = str(value).lower()
    if str_val not in valid:
        raise ValueError(f"{name} must be one of {sorted(valid)}, got {value!r}")
    return str_val


def validate_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a user-supplied training configuration dict.

    All recognised keys are validated with clear error messages; unknown keys
    are passed through unchanged so callers can embed SysPlug configs inside
    larger framework config dicts.

    Args:
        config: A dict with any subset of the following keys:
            - ``batch_size`` (positive int)
            - ``gradient_accumulation`` (positive int)
            - ``learning_rate`` (positive float)
            - ``precision`` (str, one of fp32/fp16/bf16/int8/int4)
            - ``optimizer`` (str, one of adamw/adam/sgd/adafactor)
            - ``parallelism`` (str, one of none/dp/ddp/fsdp/zero1/zero2/zero3)
            - ``use_gradient_checkpointing`` (bool)
            - ``training_type`` (str, one of supervised/sft/dpo/rlhf/grpo)
            - ``objective`` (str, one of throughput/memory/balanced)
            - ``sequence_length`` (positive int, optional)
            - ``num_train_epochs`` (positive int, optional)
            - ``max_steps`` (positive int, optional)

    Returns:
        A new dict with validated and normalised values.

    Raises:
        ValueError: On the first validation failure encountered, with a
            descriptive message.

    Examples:
        >>> validate_config_dict({"batch_size": 8, "precision": "BF16"})
        {'batch_size': 8, 'precision': 'bf16'}
        >>> validate_config_dict({"batch_size": -1})
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be a positive integer, got -1
    """
    validated: dict[str, Any] = {}

    for key, value in config.items():
        if key == "batch_size":
            validated[key] = _check_positive_int(value, "batch_size")
        elif key == "gradient_accumulation":
            validated[key] = _check_positive_int(value, "gradient_accumulation")
        elif key == "learning_rate":
            validated[key] = _check_positive_float(value, "learning_rate")
        elif key == "precision":
            validated[key] = _check_in_set(value, "precision", _VALID_PRECISION)
        elif key == "optimizer":
            validated[key] = _check_in_set(value, "optimizer", _VALID_OPTIMIZER)
        elif key == "parallelism":
            validated[key] = _check_in_set(value, "parallelism", _VALID_PARALLELISM)
        elif key == "use_gradient_checkpointing":
            validated[key] = bool(value)
        elif key == "training_type":
            validated[key] = _check_in_set(value, "training_type", _VALID_TRAINING_TYPES)
        elif key == "objective":
            validated[key] = _check_in_set(value, "objective", _VALID_OBJECTIVES)
        elif key in {"sequence_length", "num_train_epochs", "max_steps"}:
            validated[key] = _check_positive_int(value, key)
        else:
            # Pass through unknown keys unchanged
            validated[key] = value

    return validated
