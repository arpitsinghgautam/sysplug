"""Integration tests for PyTorch integration."""

from __future__ import annotations

import time
import pytest

from sysplug import Advisor
from sysplug.hardware import HardwareSnapshot
from sysplug.integrations.pytorch import SysPlugContext, SysPlugForwardHook


class TestSysPlugContext:
    def test_context_manager_basic(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4})

        with SysPlugContext(advisor, check_interval_steps=5) as ctx:
            for step in range(10):
                ctx.record(step=step, loss=1.0 - step * 0.05)

    def test_context_passes_metrics_to_monitor(
        self, mock_gpu: HardwareSnapshot
    ) -> None:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4})

        with SysPlugContext(advisor, check_interval_steps=10) as ctx:
            for step in range(20):
                ctx.record(step=step, loss=1.0, grad_norm=0.5)

    def test_context_without_suggest_still_works(
        self, mock_gpu: HardwareSnapshot
    ) -> None:
        """Context should not crash even if suggest_config wasn't called."""
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        with SysPlugContext(advisor, check_interval_steps=10) as ctx:
            ctx.record(step=0, loss=1.0)

    def test_context_with_grad_norm(self, mock_gpu: HardwareSnapshot) -> None:
        advisor = Advisor(model="gpt2", hardware=mock_gpu, verbose=False)
        advisor.suggest_config({"batch_size": 4})
        with SysPlugContext(advisor) as ctx:
            for step in range(5):
                ctx.record(step=step, loss=0.5, grad_norm=1.2)


class TestSysPlugForwardHook:
    def test_hook_creation(self) -> None:
        hook = SysPlugForwardHook()
        assert hook.last_activation_mb == 0.0

    def test_hook_callable(self) -> None:
        """Hook __call__ should not raise on CPU."""
        hook = SysPlugForwardHook(device="cpu")
        hook(None, None, None)  # no GPU, should still not crash

    def test_register_on_module(self) -> None:
        """register() should attach hooks without error."""
        pytest.importorskip("torch")
        import torch

        hook = SysPlugForwardHook()
        model = torch.nn.Linear(4, 2)
        pre, post = hook.register(model)
        # Run a forward pass
        x = torch.randn(2, 4)
        _ = model(x)
        # Clean up
        pre.remove()
        post.remove()

    def test_pre_hook_records_baseline(self) -> None:
        hook = SysPlugForwardHook()
        hook.pre_hook(None, None)  # Should not raise even without GPU
