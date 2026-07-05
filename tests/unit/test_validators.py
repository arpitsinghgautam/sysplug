"""Unit tests for validators.py."""

from __future__ import annotations

import pytest

from sysplug.utils.validators import validate_config_dict


class TestValidateConfigDict:
    def test_valid_minimal(self) -> None:
        result = validate_config_dict({"batch_size": 8})
        assert result["batch_size"] == 8

    def test_normalises_precision_uppercase(self) -> None:
        result = validate_config_dict({"precision": "BF16"})
        assert result["precision"] == "bf16"

    def test_all_valid_keys(self) -> None:
        config = {
            "batch_size": 4,
            "gradient_accumulation": 2,
            "learning_rate": 2e-5,
            "precision": "bf16",
            "optimizer": "adamw",
            "parallelism": "zero2",
            "use_gradient_checkpointing": True,
            "training_type": "sft",
            "objective": "balanced",
            "sequence_length": 512,
            "num_train_epochs": 3,
            "max_steps": 1000,
        }
        result = validate_config_dict(config)
        assert result["batch_size"] == 4
        assert result["gradient_accumulation"] == 2
        assert result["learning_rate"] == pytest.approx(2e-5)
        assert result["precision"] == "bf16"
        assert result["optimizer"] == "adamw"
        assert result["parallelism"] == "zero2"
        assert result["use_gradient_checkpointing"] is True
        assert result["training_type"] == "sft"
        assert result["objective"] == "balanced"

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            validate_config_dict({"batch_size": -1})

    def test_zero_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            validate_config_dict({"batch_size": 0})

    def test_invalid_learning_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            validate_config_dict({"learning_rate": -1e-4})

    def test_invalid_precision_raises(self) -> None:
        with pytest.raises(ValueError, match="precision"):
            validate_config_dict({"precision": "fp64"})

    def test_invalid_optimizer_raises(self) -> None:
        with pytest.raises(ValueError, match="optimizer"):
            validate_config_dict({"optimizer": "rmsprop"})

    def test_invalid_parallelism_raises(self) -> None:
        with pytest.raises(ValueError, match="parallelism"):
            validate_config_dict({"parallelism": "tensor_parallel"})

    def test_invalid_training_type_raises(self) -> None:
        with pytest.raises(ValueError, match="training_type"):
            validate_config_dict({"training_type": "ppo"})

    def test_invalid_objective_raises(self) -> None:
        with pytest.raises(ValueError, match="objective"):
            validate_config_dict({"objective": "speed"})

    def test_unknown_keys_pass_through(self) -> None:
        config = {"batch_size": 4, "custom_field": "hello", "num_workers": 4}
        result = validate_config_dict(config)
        assert result["custom_field"] == "hello"
        assert result["num_workers"] == 4

    def test_use_gradient_checkpointing_bool(self) -> None:
        result = validate_config_dict({"use_gradient_checkpointing": 1})
        assert result["use_gradient_checkpointing"] is True

    def test_empty_dict(self) -> None:
        result = validate_config_dict({})
        assert result == {}

    @pytest.mark.parametrize("prec", ["fp32", "fp16", "bf16", "int8", "int4"])
    def test_all_valid_precisions(self, prec: str) -> None:
        result = validate_config_dict({"precision": prec})
        assert result["precision"] == prec

    @pytest.mark.parametrize("opt", ["adamw", "adam", "sgd", "adafactor"])
    def test_all_valid_optimizers(self, opt: str) -> None:
        result = validate_config_dict({"optimizer": opt})
        assert result["optimizer"] == opt

    @pytest.mark.parametrize("par", ["none", "dp", "ddp", "fsdp", "zero1", "zero2", "zero3"])
    def test_all_valid_parallelism(self, par: str) -> None:
        result = validate_config_dict({"parallelism": par})
        assert result["parallelism"] == par

    def test_invalid_sequence_length_raises(self) -> None:
        with pytest.raises(ValueError, match="sequence_length"):
            validate_config_dict({"sequence_length": 0})

    def test_invalid_gradient_accumulation_raises(self) -> None:
        with pytest.raises(ValueError, match="gradient_accumulation"):
            validate_config_dict({"gradient_accumulation": -2})
