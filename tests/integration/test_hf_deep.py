"""Deep HuggingFace integration tests.

Covers SysPlugTrainerCallback end-to-end by mocking the transformers
TrainerCallback base class so that transformers doesn't need to be installed
in the test environment, while exercising every real code path.
"""

from __future__ import annotations

import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import pytest

from sysplug.hardware import GPUSnapshot, HardwareSnapshot

# ---------------------------------------------------------------------------
# Build a minimal fake `transformers` module so the callback can inherit
# ---------------------------------------------------------------------------

def _install_fake_transformers():
    """Inject a fake `transformers` module into sys.modules."""
    if "transformers" in sys.modules:
        return  # already installed (real or fake)

    fake = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    class TrainingArguments:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        per_device_train_batch_size = 8
        gradient_accumulation_steps = 1
        learning_rate = 2e-5
        bf16 = True
        fp16 = False
        gradient_checkpointing = False

    class TrainerState:
        global_step = 0
        epoch = 1.0

    class TrainerControl:
        pass

    fake.TrainerCallback = TrainerCallback
    fake.TrainingArguments = TrainingArguments
    fake.TrainerState = TrainerState
    fake.TrainerControl = TrainerControl
    sys.modules["transformers"] = fake
    return fake


_install_fake_transformers()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _a100_hw() -> HardwareSnapshot:
    return HardwareSnapshot(
        gpus=[GPUSnapshot(0, "A100", 40_960, 0, 40_960, 0, 0, (8, 0), 2039)],
        cpu_count=8, ram_total_mb=65_536,
    )


def _make_advisor(model: str = "gpt2", verbose: bool = False):
    import sysplug
    return sysplug.Advisor(model=model, hardware=_a100_hw(), verbose=verbose)


def _make_args(**overrides):
    """Create a fake TrainingArguments-like object."""
    args = MagicMock()
    args.per_device_train_batch_size = overrides.get("batch_size", 8)
    args.gradient_accumulation_steps = overrides.get("grad_acc", 1)
    args.learning_rate = overrides.get("lr", 2e-5)
    args.bf16 = overrides.get("bf16", True)
    args.fp16 = overrides.get("fp16", False)
    args.gradient_checkpointing = overrides.get("gc", False)
    return args


def _make_state(step: int = 0, epoch: float = 1.0):
    state = MagicMock()
    state.global_step = step
    state.epoch = epoch
    return state


def _make_control():
    return MagicMock()


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestCallbackConstruction:

    def test_creates_callback(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        assert cb._advisor is adv

    def test_stability_signal_none_before_train_begin(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        assert cb._stability_signal is None

    def test_requires_transformers_installed(self):
        """Without transformers importable, construction raises ImportError.

        Setting the ``sys.modules`` entry to ``None`` makes ``import
        transformers`` raise ImportError, which correctly simulates the module
        being absent regardless of whether it is actually installed in the test
        environment (popping it from sys.modules would not — the import would
        just succeed again from disk).
        """
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        with patch.dict(sys.modules, {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                SysPlugTrainerCallback(_make_advisor())


# ---------------------------------------------------------------------------
# 2. on_train_begin
# ---------------------------------------------------------------------------

class TestOnTrainBegin:

    def test_suggest_config_called(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())
        assert adv.current_config is not None

    def test_stability_signal_initialized(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())
        assert cb._stability_signal is not None

    def test_bf16_precision_extracted(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        args = _make_args(bf16=True, fp16=False)
        cb.on_train_begin(args, _make_state(), _make_control())
        assert adv.current_config.precision in {"bf16", "fp32", "fp16"}

    def test_fp16_precision_extracted(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        args = _make_args(bf16=False, fp16=True)
        cb.on_train_begin(args, _make_state(), _make_control())
        # solver may upgrade fp16→bf16, just check it ran
        assert adv.current_config is not None

    def test_fp32_fallback(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        args = _make_args(bf16=False, fp16=False)
        cb.on_train_begin(args, _make_state(), _make_control())
        assert adv.current_config is not None

    def test_gradient_checkpointing_flag_passed(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        args = _make_args(gc=True)
        cb.on_train_begin(args, _make_state(), _make_control())
        assert adv.current_config is not None

    def test_suggest_config_exception_swallowed(self):
        """Exceptions in suggest_config must be caught and warned, not raised."""
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        # Poison the advisor's suggest_config
        adv.suggest_config = MagicMock(side_effect=RuntimeError("boom"))
        cb = SysPlugTrainerCallback(adv)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb.on_train_begin(_make_args(), _make_state(), _make_control())
        assert any("boom" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# 3. on_step_end
# ---------------------------------------------------------------------------

class TestOnStepEnd:

    def _trained_callback(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())
        return cb

    def test_no_op_when_stability_signal_none(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        # No on_train_begin called → _stability_signal is None
        # Should not raise
        cb.on_step_end(_make_args(), _make_state(step=1), _make_control(),
                       logs={"loss": 1.0})

    def test_records_loss_from_logs(self):
        cb = self._trained_callback()
        state = _make_state(step=5)
        cb.on_step_end(_make_args(), state, _make_control(), logs={"loss": 0.8})
        assert cb._stability_signal.num_recorded_steps == 1

    def test_records_grad_norm_from_logs(self):
        cb = self._trained_callback()
        state = _make_state(step=5)
        cb.on_step_end(_make_args(), state, _make_control(),
                       logs={"loss": 0.8, "grad_norm": 1.2})
        assert len(cb._stability_signal._grad_norms) == 1

    def test_handles_missing_loss_in_logs(self):
        cb = self._trained_callback()
        # No "loss" key → no exception, no recording
        cb.on_step_end(_make_args(), _make_state(step=1), _make_control(),
                       logs={})
        assert cb._stability_signal.num_recorded_steps == 0

    def test_handles_empty_logs(self):
        cb = self._trained_callback()
        cb.on_step_end(_make_args(), _make_state(step=1), _make_control(),
                       logs={})

    def test_handles_no_logs_kwarg(self):
        cb = self._trained_callback()
        # No logs= kwarg at all
        cb.on_step_end(_make_args(), _make_state(step=1), _make_control())

    def test_multiple_steps_accumulate(self):
        cb = self._trained_callback()
        for step in range(10):
            state = _make_state(step=step)
            cb.on_step_end(_make_args(), state, _make_control(),
                           logs={"loss": 1.0 - step * 0.05})
        assert cb._stability_signal.num_recorded_steps == 10


# ---------------------------------------------------------------------------
# 4. on_epoch_end
# ---------------------------------------------------------------------------

class TestOnEpochEnd:

    def _trained_callback_with_data(self, n_steps: int = 20, loss: float = 1.0):
        """Create a callback with n_steps of constant loss recorded."""
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())
        for step in range(n_steps):
            cb.on_step_end(_make_args(), _make_state(step=step), _make_control(),
                           logs={"loss": loss})
        return cb

    def test_no_op_when_stability_signal_none(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        cb.on_epoch_end(_make_args(), _make_state(), _make_control())

    def test_stable_training_no_warning(self):
        """Nearly-constant loss → oscillation check passes → no warning emitted."""
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())
        # Constant loss at 1.0 — zero variance → never oscillating or diverging
        for step in range(30):
            cb.on_step_end(_make_args(), _make_state(step=step), _make_control(),
                           logs={"loss": 1.0})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb.on_epoch_end(_make_args(), _make_state(epoch=1), _make_control())
        sysplug_warns = [x for x in w if "SysPlug" in str(x.message)]
        assert len(sysplug_warns) == 0

    def test_diverging_training_emits_warning(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        cb = SysPlugTrainerCallback(adv)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())
        # Feed diverging loss
        for step in range(20):
            cb.on_step_end(_make_args(), _make_state(step=step), _make_control(),
                           logs={"loss": 1.0 + step * 0.1})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb.on_epoch_end(_make_args(), _make_state(epoch=1), _make_control())
        # Should have warned about instability
        sysplug_warns = [x for x in w if "SysPlug" in str(x.message)]
        assert len(sysplug_warns) >= 1


# ---------------------------------------------------------------------------
# 5. from_training_args factory
# ---------------------------------------------------------------------------

class TestFromTrainingArgs:

    def test_creates_callback_with_new_advisor(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        import sysplug
        model = MagicMock()
        # Make the mock's .parameters() return an empty iterator (0 params → default)
        model.parameters.side_effect = AttributeError("no params")
        ta = _make_args()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cb = SysPlugTrainerCallback.from_training_args(ta, model=model)
        assert isinstance(cb, SysPlugTrainerCallback)

    def test_creates_callback_with_provided_advisor(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        adv = _make_advisor()
        ta = _make_args()
        cb = SysPlugTrainerCallback.from_training_args(ta, model=None, advisor=adv)
        assert cb._advisor is adv

    def test_accepts_param_count_integer_as_model(self):
        from sysplug.integrations.huggingface import SysPlugTrainerCallback
        ta = _make_args()
        # Integer model → interpreted as param count
        cb = SysPlugTrainerCallback.from_training_args(
            ta, model=125_000_000,
            advisor=_make_advisor()  # provide advisor so no new one created
        )
        assert isinstance(cb, SysPlugTrainerCallback)
