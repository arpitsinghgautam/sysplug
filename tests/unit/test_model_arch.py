"""Tests for architecture introspection (resolve_model_arch / ModelArch)."""

from __future__ import annotations

import pytest

from sysplug.memory_model import ModelArch, resolve_model_arch


class _FakeParam:
    def __init__(self, n: int) -> None:
        self._n = n

    def numel(self) -> int:
        return self._n


class _FakeModule:
    """Minimal nn.Module stand-in with an optional ``.config``."""

    def __init__(self, config: object | None = None, nparams: int = 1_000_000) -> None:
        self.config = config
        self._nparams = nparams

    def parameters(self) -> list[_FakeParam]:
        return [_FakeParam(self._nparams)]


class _Cfg:
    """Bag object standing in for a HuggingFace model config."""

    def __init__(self, **kw: object) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class TestResolveFromScalar:
    def test_from_int_param_count(self) -> None:
        arch = resolve_model_arch(125_000_000)
        assert arch.param_count == 125_000_000
        assert arch.hidden_size >= 128 and arch.num_layers >= 1
        assert arch.source == "param_inference"
        assert arch.attn_impl == "eager"  # conservative default

    def test_from_string_arch_table(self) -> None:
        arch = resolve_model_arch("llama-3-8b")
        assert (arch.hidden_size, arch.num_layers, arch.num_heads) == (4096, 32, 32)
        assert arch.num_kv_heads == 8  # GQA captured
        assert arch.source == "name_table"

    def test_gpt2_medium_not_matched_as_gpt2(self) -> None:
        # Longest-key-first matching must not collapse gpt2-medium onto gpt2.
        arch = resolve_model_arch("gpt2-medium")
        assert arch.hidden_size == 1024 and arch.num_layers == 24

    def test_string_in_param_table_but_not_arch_table_falls_back(self) -> None:
        # falcon-7b has a param count but no arch-table entry -> inference, no raise.
        arch = resolve_model_arch("falcon-7b")
        assert arch.param_count == 7_000_000_000
        assert arch.hidden_size >= 128 and arch.num_layers >= 1

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model name"):
            resolve_model_arch("totally-not-a-model-xyz")


class TestResolveFromModule:
    def test_hf_config_read_directly(self) -> None:
        cfg = _Cfg(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=8192,
            _attn_implementation="flash_attention_2",
        )
        arch = resolve_model_arch(_FakeModule(config=cfg, nparams=8_000_000_000))
        assert arch.source == "config"
        assert (arch.hidden_size, arch.num_layers, arch.num_heads) == (4096, 32, 32)
        assert arch.num_kv_heads == 8
        assert arch.attn_impl == "flash_attention_2"
        assert arch.is_subquadratic_attn() is True
        assert arch.param_count == 8_000_000_000

    def test_gpt2_style_config_naming(self) -> None:
        cfg = _Cfg(n_embd=768, n_layer=12, n_head=12, n_positions=1024)
        arch = resolve_model_arch(_FakeModule(config=cfg, nparams=124_000_000))
        assert (arch.hidden_size, arch.num_layers, arch.num_heads) == (768, 12, 12)
        assert arch.num_kv_heads == 12  # defaults to query heads when absent
        assert arch.attn_impl == "eager"  # no attn field -> conservative default

    def test_plain_module_without_config_infers(self) -> None:
        arch = resolve_model_arch(_FakeModule(config=None, nparams=1_300_000_000))
        assert arch.source == "param_inference"
        assert arch.param_count == 1_300_000_000

    def test_unparseable_model_defaults_to_125m(self) -> None:
        arch = resolve_model_arch(object())  # no .parameters(), no .config
        assert arch.param_count == 125_000_000
        assert arch.source == "param_inference"


class TestSubquadraticFlag:
    @pytest.mark.parametrize(
        "impl,expected",
        [
            ("eager", False),
            ("sdpa", True),
            ("flash_attention_2", True),
            ("mem_efficient", True),
            ("", False),
        ],
    )
    def test_is_subquadratic(self, impl: str, expected: bool) -> None:
        arch = ModelArch(768, 12, 12, 12, attn_impl=impl)
        assert arch.is_subquadratic_attn() is expected
