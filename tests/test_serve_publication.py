"""_ensure publishes payload and frame atomically, as one dict entry.

The fast path reads the results cache without a lock, so a cache entry must
become visible in a single assignment — split payload/frame dicts let a
reader land between the two writes and 500 on a half-published entry. The
source assertions pin the structure (one dict, one publication site); the
roundtrip pins the behavior on the hermetic Arrow store.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from mill import serve

from conftest import ArrowStore, ns

_SERVE_SRC = Path(serve.__file__).read_text(encoding="utf-8")

_TS = [ns(2024, 1, 2, i // 4, (i % 4) * 15) for i in range(8)]


def test_no_split_cache_dicts() -> None:
    assert "_PAYLOADS" not in _SERVE_SRC
    assert "_FRAMES" not in _SERVE_SRC


def test_single_publication_site() -> None:
    assert _SERVE_SRC.count("_RESULTS[key] = (payload, result)") == 1


@pytest.fixture
def fresh_serve_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "_RESULTS", {})
    monkeypatch.setattr(serve, "_RAW_CACHE", {})
    monkeypatch.setattr(serve, "_BAR_LOCKS", {})


def test_ensure_roundtrip_on_fixture_store(
    arrow_store: ArrowStore, fresh_serve_caches: None
) -> None:
    arrow_store.write("time_15m", _TS)
    payload, frame = serve._ensure("15m_time_bars", "none", [])

    assert isinstance(payload, bytes)
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == len(_TS)
    assert json.loads(payload)["columns"] == ["datetime"]

    key = ("15m_time_bars", serve._source_version("15m_time_bars"), "none", serve._spec_hash([]))
    assert serve._RESULTS[key][0] is payload
    assert serve._RESULTS[key][1] is frame

    payload_again, frame_again = serve._ensure("15m_time_bars", "none", [])
    assert payload_again is payload
    assert frame_again is frame
