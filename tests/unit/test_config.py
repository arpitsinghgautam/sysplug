"""Unit tests for SysPlugConfig."""

from __future__ import annotations

import pytest

from sysplug.config import SysPlugConfig


class TestSysPlugConfigBasic:
    def test_default_creation(self) -> None:
        cfg = SysPlugConfig()
        assert cfg.batch_size == 8
        assert cfg.learning_rate == pytest.approx(1e-4)
        assert cfg.precision == "bf16"

    def test_to_dict_contains_all_fields(self) -> None:
        cfg = SysPlugConfig(batch_size=4, learning_rate=2e-5)
        d = cfg.to_dict()
        assert d["batch_size"] == 4
        assert d["learning_rate"] == pytest.approx(2e-5)
        assert "effective_batch_size" in d
        assert "predicted_peak_memory_mb" in d
        assert "warnings" in d
        assert "notes" in d

    def test_repr(self) -> None:
        cfg = SysPlugConfig(batch_size=4)
        r = repr(cfg)
        assert "SysPlugConfig" in r
        assert "4" in r

    def test_summary_verbose(self) -> None:
        cfg = SysPlugConfig(batch_size=4, learning_rate=2e-5)
        s = cfg.summary(verbose=True)
        assert "batch_size" in s
        assert "SysPlug" in s

    def test_summary_non_verbose(self) -> None:
        cfg = SysPlugConfig(batch_size=4)
        s = cfg.summary(verbose=False)
        assert "batch_size" in s.lower() or "batch=" in s.lower()

    def test_summary_with_warnings(self) -> None:
        cfg = SysPlugConfig(warnings=["Test warning"])
        s = cfg.summary()
        assert "Test warning" in s

    def test_to_deepspeed_config_basic(self) -> None:
        cfg = SysPlugConfig(
            batch_size=4,
            gradient_accumulation=2,
            precision="bf16",
            gpu_count=1,
            parallelism="zero2",
        )
        ds = cfg.to_deepspeed_config()
        assert ds["train_micro_batch_size_per_gpu"] == 4
        assert ds["gradient_accumulation_steps"] == 2
        assert ds["bf16"]["enabled"] is True
        assert ds["zero_optimization"]["stage"] == 2

    def test_to_deepspeed_config_fp16(self) -> None:
        cfg = SysPlugConfig(precision="fp16", parallelism="none")
        ds = cfg.to_deepspeed_config()
        assert ds["fp16"]["enabled"] is True
        assert "bf16" not in ds

    def test_to_deepspeed_config_fp32(self) -> None:
        cfg = SysPlugConfig(precision="fp32", parallelism="none")
        ds = cfg.to_deepspeed_config()
        assert "fp16" not in ds
        assert "bf16" not in ds

    def test_to_deepspeed_config_with_base(self) -> None:
        cfg = SysPlugConfig(batch_size=4, precision="bf16", parallelism="none")
        base = {"gradient_clipping": 1.0}
        ds = cfg.to_deepspeed_config(base)
        assert ds["gradient_clipping"] == 1.0
        assert ds["train_micro_batch_size_per_gpu"] == 4

    def test_apply_to_optimizer(self) -> None:
        pytest.importorskip("torch")
        import torch

        model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cfg = SysPlugConfig(learning_rate=2e-5)
        cfg.apply_to_optimizer(opt)
        assert opt.param_groups[0]["lr"] == pytest.approx(2e-5)

    def test_to_training_arguments_raises_without_transformers(self) -> None:
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "transformers":
                raise ImportError("No transformers")
            return real_import(name, *args, **kwargs)

        import unittest.mock as mock

        cfg = SysPlugConfig(batch_size=4)
        with mock.patch("builtins.__import__", mock_import):
            with pytest.raises(ImportError, match="transformers"):
                cfg.to_training_arguments(output_dir="/tmp")

    def test_to_training_arguments_with_transformers(self) -> None:
        pytest.importorskip("transformers")
        cfg = SysPlugConfig(
            batch_size=4, learning_rate=2e-5, precision="bf16", use_gradient_checkpointing=False
        )
        ta = cfg.to_training_arguments(output_dir="/tmp/test")
        assert ta.per_device_train_batch_size == 4
        assert ta.learning_rate == pytest.approx(2e-5)

    def test_effective_batch_size_field(self) -> None:
        cfg = SysPlugConfig(batch_size=4, gradient_accumulation=2)
        # effective_batch_size is just a stored field, not auto-computed
        cfg.effective_batch_size = 8
        assert cfg.effective_batch_size == 8


class TestSysPlugConfigZeroStages:
    @pytest.mark.parametrize(
        "parallelism,expected_stage",
        [
            ("zero1", 1),
            ("zero2", 2),
            ("zero3", 3),
            ("fsdp", 3),
            ("none", 0),
            ("ddp", 0),
        ],
    )
    def test_zero_stage_mapping(self, parallelism: str, expected_stage: int) -> None:
        cfg = SysPlugConfig(parallelism=parallelism)
        ds = cfg.to_deepspeed_config()
        if expected_stage > 0:
            assert ds.get("zero_optimization", {}).get("stage") == expected_stage
        else:
            assert ds.get("zero_optimization", {}).get("stage") is None
