"""Mill serves every bar the store holds, through the latest closed one, and its caches follow the store as bars close.

Oracles: bar counts and days written to the hermetic Arrow store of conftest,
and the store file's modification time, set as the store's writes would set it.
"""

from __future__ import annotations

import base64
import os
import threading
import time
import urllib.error
import urllib.request
from mill.http_server import BoundedHTTPServer as ThreadingHTTPServer

import numpy as np
import pytest

from mill import discover, serve
from mill.get_data import GetData
from mill.settings import arrow_latest_path

from conftest import ArrowStore, ns

_H = 3_600_000_000_000
# Twenty hours of 2026-10-15, after where Mill once stopped reading.
_TS = [ns(2026, 10, 15) + i * _H for i in range(20)]


def _later(series: str) -> None:
    """Move the store's file a second on, as its next write would, past any clock granularity."""
    path = arrow_latest_path(series)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def _written(series: str, moment_ns: int) -> None:
    """Stamp the store's file as written at that moment."""
    os.utime(arrow_latest_path(series), ns=(moment_ns, moment_ns))


def test_get_data_serves_the_bars_closed_when_the_store_wrote(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    # Written at 19:30, the 19:00 bar was still open; the store writes it as it fills.
    _written("time_1h", _TS[-1] + _H // 2)
    assert GetData().get_time_bars("1h_time_bars").height == 19
    # Written again once it closed.
    _written("time_1h", _TS[-1] + _H)
    assert GetData().get_time_bars("1h_time_bars").height == 20


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
    assert serve._data_window("1h_time_bars") == {"start": "2026-10-15", "end": "2026-10-15"}
    # The store writes a bar of the next day.
    arrow_store.write("time_1h", [*_TS, _TS[-1] + 5 * _H])
    _later("time_1h")
    _, frame = serve._ensure("1h_time_bars", "none", [])
    assert frame.height == 21
    assert len(serve._RESULTS) == 1, "the frame of the earlier version is forgotten"
    assert serve._data_window("1h_time_bars") == {"start": "2026-10-15", "end": "2026-10-16"}


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


def test_a_rewrite_of_the_open_bar_alone_keeps_what_was_computed(
    arrow_store: ArrowStore, fresh_caches: None
) -> None:
    written = arrow_store.write("time_1h", _TS)
    _written("time_1h", _TS[-1] + _H // 2)
    _, frame = serve._ensure("1h_time_bars", "none", [])
    version = serve._raw_for("1h_time_bars")[1]
    # A minute later the store rewrites the file with the open 19:00 bar's new close.
    close = written["close"].scatter(len(_TS) - 1, -1.0)
    arrow_store.write("time_1h", _TS, extra={"close": close})
    _written("time_1h", _TS[-1] + _H // 2 + 60 * 1_000_000_000)
    assert serve._raw_for("1h_time_bars")[1] == version
    assert serve._ensure("1h_time_bars", "none", [])[1] is frame


def _pool(raw, pause=None, **kwargs) -> discover.Pool:
    """A pool without columns over the bars, for the server's bookkeeping alone."""
    return discover.Pool(names=(), natives=(), factories=(), sources=(), params=(),
                         ms=raw["datetime"].dt.epoch("ms").to_numpy(), values=np.empty((raw.height, 0)), skipped=())


def test_the_last_pool_answers_while_the_next_one_builds(
    arrow_store: ArrowStore, fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, release = [], threading.Event()

    def catalog_pool(raw, pause=None, **kwargs):
        if built:
            assert release.wait(5), "the test never released the second build"
        built.append(raw.height)
        return _pool(raw)

    monkeypatch.setattr(discover, "catalog_pool", catalog_pool)
    arrow_store.write("time_1h", _TS)
    first, cached, _ = serve._pool_for("1h_time_bars", {"kind": "catalog"})
    assert not cached and first.ms.size == 20
    # A bar closes: the first pool answers at once while the next one builds behind it.
    arrow_store.write("time_1h", [*_TS, _TS[-1] + _H])
    _later("time_1h")
    pool, cached, _ = serve._pool_for("1h_time_bars", {"kind": "catalog"})
    assert pool is first and cached
    # Searches meanwhile answer from the first pool too, without starting more builds.
    threads = threading.active_count()
    for _ in range(5):
        assert serve._pool_for("1h_time_bars", {"kind": "catalog"})[0] is first
    assert threading.active_count() == threads
    release.set()
    deadline = time.monotonic() + 5
    while len(serve._POOLS) != 1 or next(iter(serve._POOLS.values())) is first:
        assert time.monotonic() < deadline, "the next pool never replaced the first"
        time.sleep(0.01)
    pool, cached, _ = serve._pool_for("1h_time_bars", {"kind": "catalog"})
    assert cached and pool.ms.size == 21 and built == [20, 21]


def test_warming_builds_each_time_bars_pool_fewest_bars_first(
    arrow_store: ArrowStore, fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = []

    def catalog_pool(raw, pause=None, **kwargs):
        built.append(raw.height)
        return _pool(raw)

    monkeypatch.setattr(discover, "catalog_pool", catalog_pool)
    arrow_store.write("time_1h", _TS)
    arrow_store.write("time_4h", _TS[::4])
    # The store holds two of the five time bars; the others are left to their first search.
    serve._warm()
    assert built == [5, 20]
    assert sorted(key[0] for key in serve._POOLS) == ["1h_time_bars", "4h_time_bars"]


def test_view_pools_never_evict_a_time_bars_catalog_pool(
    arrow_store: ArrowStore, fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(discover, "catalog_pool", _pool)
    monkeypatch.setattr(discover, "spec_pool", lambda raw, spec, pause=None, **kwargs: _pool(raw))
    monkeypatch.setattr(serve, "_VIEW_SPECS", {f"v{i}": [{"factory": "close", "params": {}}] for i in range(5)})
    arrow_store.write("time_1h", _TS)
    serve._pool_for("1h_time_bars", {"kind": "catalog"})
    for i in range(5):
        serve._pool_for("1h_time_bars", {"kind": "view", "name": f"v{i}"})
    # The catalog pool stays; the three most recent views are kept beside it.
    assert [key[2].rsplit(":", 1)[0] for key in serve._POOLS] == ["catalog", "view:v2", "view:v3", "view:v4"]


_SPEC = [{"factory": "close", "params": {}}, {"factory": "volume", "params": {}}]


def _refreshing(key: tuple, raw, spec: list = _SPEC) -> tuple[threading.Thread, threading.Event]:
    """A refresh of key in its own thread, its lock held for it as _pool_for holds it."""
    lock, done = threading.Lock(), threading.Event()
    lock.acquire()
    worker = threading.Thread(target=lambda: (serve._refresh_pool(key, lock, raw, spec), done.set()))
    worker.start()
    return worker, done


def test_a_refresh_waits_between_factories_while_a_search_runs(arrow_store: ArrowStore, fresh_caches: None) -> None:
    arrow_store.write("time_1h", _TS)
    raw, version = serve._raw_for("1h_time_bars")
    key = ("1h_time_bars", version, "view:ochlv:x")
    with serve._search():
        worker, done = _refreshing(key, raw)
        assert not done.wait(0.3), "a refresh must wait while a search runs"
    worker.join(5)
    assert done.is_set() and serve._POOLS[key].names == ("close", "volume")


def test_a_refresh_waits_only_for_the_searches_running_when_it_paused(
    arrow_store: ArrowStore, fresh_caches: None
) -> None:
    arrow_store.write("time_1h", _TS)
    raw, version = serve._raw_for("1h_time_bars")
    first = serve._search()
    first.__enter__()
    # One factory, so one pause: the refresh waits there for the first search alone.
    worker, done = _refreshing(("1h_time_bars", version, "view:close:x"), raw, _SPEC[:1])
    assert not done.wait(0.2)
    # A later search starts while the refresh waits; the first one ends.
    with serve._search():
        first.__exit__(None, None, None)
        assert done.wait(5), "a search that began later must not hold the refresh"
    worker.join(5)


def test_a_first_build_does_not_wait_for_searches(
    arrow_store: ArrowStore, fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serve, "_VIEW_SPECS", {"ochlv": _SPEC})
    arrow_store.write("time_1h", _TS)
    done = threading.Event()
    with serve._search():
        worker = threading.Thread(target=lambda: (serve._pool_for("1h_time_bars", {"kind": "view", "name": "ochlv"}), done.set()))
        worker.start()
        assert done.wait(5), "a request waiting for a pool's first build must not wait for other searches"
    worker.join(5)


def test_a_search_is_counted_while_it_runs(
    arrow_store: ArrowStore, fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    monkeypatch.setattr(discover, "discover", lambda pool, **kw: seen.append(len(serve._SEARCHING)) or {})
    monkeypatch.setattr(serve, "_VIEW_SPECS", {"c": [{"factory": "close", "params": {}}]})
    arrow_store.write("time_1h", _TS)
    serve._discover({"pool": {"kind": "view", "name": "c"},
                     "windows": [{"start": "2026-10-15T02:00:00Z", "end": "2026-10-15T04:00:00Z"}]})
    assert seen == [1] and serve._SEARCHING == set()


def test_retiring_waits_for_the_registry_lock(fresh_caches: None) -> None:
    serve._RESULTS[("1h_time_bars", 1, "none", "")] = (b"", None)
    done = threading.Event()
    with serve._REGISTRY_LOCK:
        worker = threading.Thread(target=lambda: (serve._retire("1h_time_bars", 2), done.set()))
        worker.start()
        assert not done.wait(0.2), "retiring must not mutate the caches while another holder of the lock does"
    worker.join(5)
    assert done.is_set() and serve._RESULTS == {}


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
