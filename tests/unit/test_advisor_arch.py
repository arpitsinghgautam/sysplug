"""Advisor introspects and stores a ModelArch (not the nn.Module)."""

from __future__ import annotations

import sysplug
from sysplug.hardware import GPUSnapshot, HardwareSnapshot
from sysplug.memory_model import ModelArch


def _hw() -> HardwareSnapshot:
    return HardwareSnapshot(
        gpus=[GPUSnapshot(0, "A100", 40_960, 0, 40_960, 0, 0, (8, 0), 2039)],
        cpu_count=8,
        ram_total_mb=65_536,
    )


class _FakeParam:
    def __init__(self, n: int) -> None:
        self._n = n

    def numel(self) -> int:
        return self._n


class _Cfg:
    hidden_size = 4096
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    _attn_implementation = "eager"


class _FakeModule:
    config = _Cfg()

    def parameters(self) -> list[_FakeParam]:
        return [_FakeParam(8_000_000_000)]


def test_advisor_stores_arch_from_module_config() -> None:
    adv = sysplug.Advisor(model=_FakeModule(), hardware=_hw(), verbose=False)
    assert isinstance(adv._arch, ModelArch)
    assert adv._arch.source == "config"
    assert adv._arch.hidden_size == 4096
    assert adv._arch.num_layers == 32
    assert adv._arch.num_kv_heads == 8  # GQA captured
    assert adv._param_count == 8_000_000_000


def test_advisor_arch_from_name() -> None:
    adv = sysplug.Advisor(model="llama-3-8b", hardware=_hw(), verbose=False)
    assert adv._arch.hidden_size == 4096
    assert adv._arch.num_kv_heads == 8


def test_suggest_reports_conservative_upper_bound() -> None:
    adv = sysplug.Advisor(model=_FakeModule(), hardware=_hw(), verbose=False)
    cfg = adv.suggest_config({"batch_size": 1, "learning_rate": 1e-4})
    assert cfg.predicted_peak_memory_upper_mb >= cfg.predicted_peak_memory_mb > 0
