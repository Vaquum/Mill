"""Mill serves every bar the store holds, through the latest, and its caches follow the store as bars arrive.

Oracles: bar counts and days written to the hermetic Arrow store of conftest,
and the store file's modification time, moved on as a new write would move it.
"""

from __future__ import annotations

import base64
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

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


def test_a_store_write_during_a_read_leaves_the_cache_stale(arrow_store: ArrowStore, monkeypatch: pytest.MonkeyPatch) -> None:
    arrow_store.write("time_1h", _TS)
    read = GetData._load_arrow

    def racing(self, series, start, end):
        frame = read(self, series, start, end)
        # The store writes a new bar after the read and before the cache is written.
        arrow_store.write("time_1h", [*_TS, _TS[-1] + _H])
        return frame

    monkeypatch.setattr(GetData, "_load_arrow", racing)
    assert GetData().get_time_bars("1h_time_bars").height == 20
    monkeypatch.setattr(GetData, "_load_arrow", read)
    assert GetData().get_time_bars("1h_time_bars").height == 21


def test_retiring_waits_for_the_registry_lock_that_pool_eviction_holds(fresh_caches: None) -> None:
    serve._POOLS[("1h_time_bars", 1, "catalog")] = object()
    done = threading.Event()
    with serve._REGISTRY_LOCK:
        worker = threading.Thread(target=lambda: (serve._retire("1h_time_bars", 2), done.set()))
        worker.start()
        assert not done.wait(0.2), "retiring must not mutate the pools while another holder of the lock evicts"
    worker.join(5)
    assert done.is_set() and serve._POOLS == {}


@pytest.fixture
def server(arrow_store: ArrowStore, fresh_caches: None, monkeypatch: pytest.MonkeyPatch):
    for name, value in (("_APP_USERNAME", "u"), ("_APP_PASSWORD", "p"), ("_API_SECRET", "s")):
        monkeypatch.setattr(serve, name, value)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve._Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        thread.join(timeout=5)


def test_a_qc_report_for_a_series_the_store_lacks_fails_as_an_answer(server: str) -> None:
    request = urllib.request.Request(server + "/qc?bars=1h_time_bars&target=none&features=%5B%5D",
                                     headers={"Authorization": "Basic " + base64.b64encode(b"u:p").decode()})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    with error.value as answer:
        assert answer.code == 500 and answer.read().startswith(b"qc failed")
