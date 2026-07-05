"""Integration tests for Hugging Face callback."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from sysplug import Advisor, SysPlugConfig
from sysplug.hardware import HardwareSnapshot


class TestSysPlugTrainerCallback:
    @pytest.fixture
    def advisor(self, mock_gpu: HardwareSnapshot) -> Advisor:
        return Advisor(model="gpt2", hardware=mock_gpu, verbose=False)

    @pytest.fixture
    def mock_training_args(self) -> MagicMock:
        args = MagicMock()
        args.per_device_train_batch_size = 4
        args.gradient_accumulation_steps = 1
        args.learning_rate = 2e-5
        args.bf16 = True
        args.fp16 = False
        args.gradient_checkpointing = False
        return args

    @pytest.fixture
    def mock_state(self) -> MagicMock:
        state = MagicMock()
        state.global_step = 0
        state.epoch = 1.0
        return state

    @pytest.fixture
    def mock_control(self) -> MagicMock:
        return MagicMock()

    def test_import_raises_without_transformers(self) -> None:
        """Should raise ImportError when transformers not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(ImportError, match="transformers"):
                from sysplug.integrations.huggingface import (
                    _require_transformers,
                )
                _require_transformers()

    def test_callback_on_train_begin_calls_suggest(
        self,
        advisor: Advisor,
        mock_training_args: MagicMock,
        mock_state: MagicMock,
        mock_control: MagicMock,
    ) -> None:
        """on_train_begin should call suggest_config."""
        pytest.importorskip("transformers")
        from sysplug.integrations.huggingface import SysPlugTrainerCallback

        cb = SysPlugTrainerCallback(advisor)
        cb.on_train_begin(mock_training_args, mock_state, mock_control)
        # Advisor should have a current config after suggest
        assert advisor.current_config is not None

    def test_callback_records_loss(
        self,
        advisor: Advisor,
        mock_training_args: MagicMock,
        mock_state: MagicMock,
        mock_control: MagicMock,
    ) -> None:
        """on_step_end should record loss without error."""
        pytest.importorskip("transformers")
        from sysplug.integrations.huggingface import SysPlugTrainerCallback

        cb = SysPlugTrainerCallback(advisor)
        cb.on_train_begin(mock_training_args, mock_state, mock_control)
        mock_state.global_step = 10
        cb.on_step_end(
            mock_training_args,
            mock_state,
            mock_control,
            logs={"loss": 0.5},
        )
        # Stability signal should have the loss recorded
        assert cb._stability_signal is not None
        assert cb._stability_signal.num_recorded_steps >= 1

    def test_to_training_arguments_sets_correct_fields(
        self, mock_gpu: HardwareSnapshot
    ) -> None:
        """to_training_arguments should set batch_size and lr."""
        pytest.importorskip("transformers")
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        cfg = advisor.suggest_config({"batch_size": 4, "learning_rate": 2e-5})
        ta = cfg.to_training_arguments(output_dir="/tmp/test_model")
        assert ta.per_device_train_batch_size == cfg.batch_size
        assert ta.learning_rate == pytest.approx(cfg.learning_rate)

    def test_from_training_args_creates_callback(
        self,
        mock_gpu: HardwareSnapshot,
    ) -> None:
        pytest.importorskip("transformers")
        from sysplug.integrations.huggingface import SysPlugTrainerCallback

        mock_ta = MagicMock()
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])

        # with advisor explicitly provided
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        cb = SysPlugTrainerCallback.from_training_args(mock_ta, mock_model, advisor)
        assert cb._advisor is advisor
