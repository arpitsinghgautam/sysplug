"""Pure PyTorch integration for SysPlug.

Provides a context manager and a forward hook for manual training loops:

    with SysPlugContext(advisor, check_interval_steps=100) as ctx:
        for step, batch in enumerate(dataloader):
            with optimizer.zero_grad():
                loss = model(batch)
                loss.backward()
            ctx.record(step=step, loss=loss.item())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sysplug.advisor import Advisor
    from sysplug.monitor import Monitor


class SysPlugContext:
    """Context manager wrapping a raw PyTorch training loop.

    A thin wrapper around :class:`~sysplug.monitor.Monitor` that is
    designed for use in pure PyTorch training loops without Hugging Face
    Trainer.

    Args:
        advisor: A configured :class:`~sysplug.advisor.Advisor`.
        check_interval_steps: How often to run GPU + stability checks.
        reconfig_policy: ``"suggest"``, ``"auto-apply"``, or ``"warn-only"``.

    Examples:
        >>> import sysplug
        >>> advisor = sysplug.Advisor(model="gpt2")
        >>> _ = advisor.suggest_config({"batch_size": 4})
        >>> with SysPlugContext(advisor, check_interval_steps=10) as ctx:
        ...     for step in range(20):
        ...         ctx.record(step=step, loss=1.0 - step * 0.01)
    """

    def __init__(
        self,
        advisor: "Advisor",
        check_interval_steps: int = 50,
        reconfig_policy: str = "suggest",
    ) -> None:
        self._advisor = advisor
        self._check_interval = check_interval_steps
        self._reconfig_policy = reconfig_policy
        self._monitor: Optional["Monitor"] = None

    def __enter__(self) -> "SysPlugContext":
        self._monitor = self._advisor.monitor(
            check_interval_steps=self._check_interval,
            reconfig_policy=self._reconfig_policy,
        ).__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._monitor is not None:
            self._monitor.__exit__(*args)

    def record(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        custom_metrics: Optional[Any] = None,
    ) -> None:
        """Record a training step (thread-safe, non-blocking).

        Args:
            step: Global training step index.
            loss: Training loss value.
            grad_norm: Optional gradient L2-norm.
            custom_metrics: Optional dict of additional metrics.
        """
        if self._monitor is not None:
            self._monitor.record(
                step=step,
                loss=loss,
                grad_norm=grad_norm,
                custom_metrics=custom_metrics,
            )


class SysPlugForwardHook:
    """``nn.Module`` forward hook that measures per-step activation memory.

    Attach to any module to track how much CUDA memory is consumed by
    its forward pass activations.

    Args:
        device: CUDA device to measure (default: ``"cuda:0"``).

    Examples:
        >>> import torch
        >>> hook = SysPlugForwardHook()
        >>> model = torch.nn.Linear(64, 32)
        >>> handle = model.register_forward_hook(hook)
        >>> _ = model(torch.randn(4, 64))
        >>> hook.last_activation_mb
        0.0  # no GPU in test environment
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self._device = device
        self._before_bytes: int = 0
        self.last_activation_mb: float = 0.0

    def __call__(
        self,
        module: Any,
        input: Any,
        output: Any,
    ) -> None:
        """Called after each forward pass of the monitored module."""
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                after_bytes = torch.cuda.memory_allocated(self._device)
                self.last_activation_mb = (
                    after_bytes - self._before_bytes
                ) / 1024 / 1024
        except Exception:
            self.last_activation_mb = 0.0

    def pre_hook(self, module: Any, input: Any) -> None:
        """Pre-forward hook to record baseline memory."""
        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                self._before_bytes = torch.cuda.memory_allocated(self._device)
        except Exception:
            self._before_bytes = 0

    def register(self, module: Any) -> tuple[Any, Any]:
        """Register both pre and post hooks on a module.

        Args:
            module: An ``nn.Module`` instance.

        Returns:
            Tuple of ``(pre_handle, post_handle)`` for later removal.
        """
        pre = module.register_forward_pre_hook(self.pre_hook)
        post = module.register_forward_hook(self)
        return pre, post
