"""Hugging Face Transformers integration for SysPlug.

Provides :class:`SysPlugTrainerCallback` which hooks into the Hugging Face
``Trainer`` to automatically suggest configurations and monitor stability.

Requires ``pip install sysplug[hf]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from sysplug.advisor import Advisor
    from sysplug.config import SysPlugConfig


def _require_transformers() -> Any:
    """Lazily import transformers with a helpful error message."""
    try:
        import transformers  # type: ignore[import]
        return transformers
    except ImportError:
        raise ImportError(
            "transformers is required for SysPlugTrainerCallback. "
            "Install it with: pip install sysplug[hf]"
        )


class SysPlugTrainerCallback:
    """Hugging Face ``TrainerCallback`` that integrates SysPlug.

    Hooks into the training lifecycle to:
    - Suggest an optimised config at the start of training.
    - Record loss at each step for stability monitoring.
    - Check stability at the end of each epoch.

    Args:
        advisor: A configured :class:`~sysplug.advisor.Advisor` instance.

    Examples:
        >>> from sysplug.integrations.huggingface import SysPlugTrainerCallback
        >>> import sysplug
        >>> advisor = sysplug.Advisor(model="gpt2", training_type="sft")
        >>> callback = SysPlugTrainerCallback(advisor)
        >>> # trainer = Trainer(..., callbacks=[callback])
    """

    def __init__(self, advisor: "Advisor") -> None:
        transformers = _require_transformers()
        # Inherit from TrainerCallback at runtime to avoid top-level import
        self.__class__ = type(
            "SysPlugTrainerCallback",
            (transformers.TrainerCallback, SysPlugTrainerCallback),
            dict(SysPlugTrainerCallback.__dict__),
        )
        self._advisor = advisor
        self._stability_signal: Optional[Any] = None

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Call ``advisor.suggest_config`` from the training arguments.

        Args:
            args: ``TrainingArguments`` instance.
            state: ``TrainerState`` instance.
            control: ``TrainerControl`` instance.
            **kwargs: Additional keyword arguments (ignored).
        """
        from sysplug.stability import StabilitySignal

        config_dict = {
            "batch_size": getattr(args, "per_device_train_batch_size", 8),
            "gradient_accumulation": getattr(args, "gradient_accumulation_steps", 1),
            "learning_rate": getattr(args, "learning_rate", 1e-4),
            "precision": (
                "bf16" if getattr(args, "bf16", False)
                else "fp16" if getattr(args, "fp16", False)
                else "fp32"
            ),
            "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
        }

        try:
            cfg = self._advisor.suggest_config(config_dict)
            self._last_config: "SysPlugConfig" = cfg
        except Exception as e:
            import warnings
            warnings.warn(f"SysPlug suggest_config failed: {e}")

        self._stability_signal = StabilitySignal(window_size=50)

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Record training loss and check for instability.

        Args:
            args: ``TrainingArguments`` instance.
            state: ``TrainerState`` instance.
            control: ``TrainerControl`` instance.
            **kwargs: May contain ``logs`` dict with the current loss.
        """
        if self._stability_signal is None:
            return

        logs: Dict[str, Any] = kwargs.get("logs", {})
        loss = logs.get("loss")
        if loss is not None:
            self._stability_signal.record_loss(state.global_step, float(loss))

        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            self._stability_signal.record_grad_norm(state.global_step, float(grad_norm))

    def on_epoch_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Check stability signal at the end of each epoch.

        Emits a warning via ``trainer.log`` if instability is detected.

        Args:
            args: ``TrainingArguments`` instance.
            state: ``TrainerState`` instance.
            control: ``TrainerControl`` instance.
            **kwargs: Additional keyword arguments.
        """
        if self._stability_signal is None:
            return

        report = self._stability_signal.check()
        if report.recommended_action != "ok":
            import warnings
            warnings.warn(
                f"[SysPlug] Epoch {state.epoch:.0f} stability check: "
                f"{report.message} (action={report.recommended_action})"
            )

    @classmethod
    def from_training_args(
        cls,
        training_args: Any,
        model: Any,
        advisor: Optional["Advisor"] = None,
    ) -> "SysPlugTrainerCallback":
        """Create a callback from an existing ``TrainingArguments`` and model.

        Args:
            training_args: A ``transformers.TrainingArguments`` instance.
            model: The ``nn.Module`` being trained.
            advisor: An existing :class:`~sysplug.advisor.Advisor` to use.
                If ``None``, a new one is created.

        Returns:
            A configured :class:`SysPlugTrainerCallback`.
        """
        if advisor is None:
            import sysplug
            advisor = sysplug.Advisor(model=model)
        return cls(advisor)
