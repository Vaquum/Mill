"""Mill serves every bar the store holds, through the latest, and its caches follow the store as bars arrive.

Oracles: bar counts and days written to the hermetic Arrow store of conftest,
and the store file's modification time, moved on as a new write would move it.
"""

from __future__ import annotations

import os

import pytest

from mill import serve
from mill.get_data import GetData
from mill.settings import arrow_latest_path

from conftest import ArrowStore, ns

_H = 3_600_000_000_000
# Twenty hours of 2026-10-16, after where Mill once stopped reading.
_TS = [ns(2026, 10, 16) + i * _H for i in range(20)]


def _later(series: str) -> None:
    """Move the store's file a second on, as its next write would, past any clock granularity."""
    path = arrow_latest_path(series)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def test_get_data_follows_the_store_as_bars_arrive(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    assert GetData().get_time_bars("1h_time_bars").height == 20
    arrow_store.write("time_1h", [*_TS, _TS[-1] + _H])
    _later("time_1h")
    assert GetData().get_time_bars("1h_time_bars").height == 21


@pytest.fixture
def fresh_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_RESULTS", "_RAW_CACHE", "_BAR_LOCKS", "_QC_CACHE", "_QC_LOCKS", "_POOLS", "_POOL_LOCKS"):
        monkeypatch.setattr(serve, name, {})


def test_the_server_reads_new_bars_and_forgets_what_earlier_ones_gave(
    arrow_store: ArrowStore, fresh_caches: None
) -> None:
    arrow_store.write("time_1h", _TS)
    _, frame = serve._ensure("1h_time_bars", "none", [])
    assert frame.height == 20 and len(serve._RESULTS) == 1
    assert serve._data_window("1h_time_bars") == {"start": "2026-10-16", "end": "2026-10-16"}
    # The store writes a bar of the next day.
    arrow_store.write("time_1h", [*_TS, _TS[-1] + 5 * _H])
    _later("time_1h")
    _, frame = serve._ensure("1h_time_bars", "none", [])
    assert frame.height == 21
    assert len(serve._RESULTS) == 1, "the frame of the earlier version is forgotten"
    assert serve._data_window("1h_time_bars") == {"start": "2026-10-16", "end": "2026-10-17"}
