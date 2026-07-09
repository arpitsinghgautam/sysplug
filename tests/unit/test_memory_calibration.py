"""Validate the calibrated memory model against the committed measurements.

Uses ``paper/data/gpu_measurements.json`` (real peak VRAM measured on a GPU).
The measured GPT-2 runs used memory-efficient SDPA attention, so predictions
here use ``attn_impl="sdpa"`` to match the code path that produced the numbers.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from sysplug.memory_model import MemoryModel

_JSON = Path(__file__).resolve().parents[2] / "paper" / "data" / "gpu_measurements.json"
_MEASURED_ATTN = "sdpa"  # GPT2LMHeadModel default in current transformers


def _ok_points() -> tuple[dict, list[dict]]:
    if not _JSON.exists():
        pytest.skip("no committed measurement data")
    data = json.loads(_JSON.read_text())
    ok = [m for m in data["measurements"] if m["ok"]]
    if not ok:
        pytest.skip("no successful measurements in dataset")
    return data, ok


def _predict(mm: MemoryModel, data: dict, m: dict) -> object:
    return mm.predict(
        param_count=m["param_count"],
        batch_size=m["batch_size"],
        precision=data["precision"],
        optimizer="adamw",
        sequence_length=data["seq_len"],
        hidden_dim=m["hidden_size"],
        num_layers=m["num_layers"],
        attn_impl=_MEASURED_ATTN,
    )


def test_conservative_upper_bound_covers_measured() -> None:
    """The OOM-safe upper bound must cover every measured peak (the (b) promise)."""
    data, ok = _ok_points()
    mm = MemoryModel(gpu_count=1)
    uncovered = [
        (
            m["config"],
            m["batch_size"],
            round(_predict(mm, data, m).upper_mb),
            m["peak_mib_allocated"],
        )
        for m in ok
        if _predict(mm, data, m).upper_mb < m["peak_mib_allocated"]
    ]
    assert not uncovered, f"conservative upper bound under-covered points: {uncovered}"


def test_central_mape_reasonable() -> None:
    data, ok = _ok_points()
    mm = MemoryModel(gpu_count=1)
    errs = [
        abs(_predict(mm, data, m).peak_memory_mb - m["peak_mib_allocated"])
        / m["peak_mib_allocated"]
        for m in ok
    ]
    assert statistics.mean(errs) < 0.20  # calibrated central MAPE
