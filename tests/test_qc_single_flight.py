"""Identical concurrent /qc requests compute once; distinct keys compute.

The /qc cache had no in-flight tracking: two users (or one double-click)
each paid the full report cost on the heaviest endpoint. The per-key lock
makes the first request compute and identical concurrent requests wait and
serve the cached body.

Hermetic: synthetic Arrow store, qc.report wrapped (not replaced) with a
counting, sleeping delegate so overlap is guaranteed and bodies stay real.
"""

from __future__ import annotations

import base64
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from mill import qc, serve

from conftest import ArrowStore, ns

_TS = [ns(2024, 1, 2, i // 4, (i % 4) * 15) for i in range(8)]
_QC_QUERY = "/qc?bars=15m_time_bars&target=none&features=%5B%5D&row_mode=bar"
_AUTH = "Basic " + base64.b64encode(b"u:p").decode()


@pytest.fixture
def server(arrow_store: ArrowStore, monkeypatch: pytest.MonkeyPatch):
    arrow_store.write("time_15m", _TS)
    monkeypatch.setattr(serve, "_RESULTS", {})
    monkeypatch.setattr(serve, "_RAW_CACHE", {})
    monkeypatch.setattr(serve, "_BAR_LOCKS", {})
    monkeypatch.setattr(serve, "_QC_CACHE", {})
    monkeypatch.setattr(serve, "_QC_LOCKS", {})
    monkeypatch.setattr(serve, "_APP_USERNAME", "u")
    monkeypatch.setattr(serve, "_APP_PASSWORD", "p")
    monkeypatch.setattr(serve, "_API_SECRET", "s")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve._Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"Authorization": _AUTH})
    with urllib.request.urlopen(request) as resp:
        return resp.status, resp.read()


def test_identical_concurrent_qc_computes_once(
    server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_report = qc.report

    def slow_counting_report(df, target):
        calls.append(1)
        time.sleep(0.3)
        return real_report(df, target)

    monkeypatch.setattr(serve.qc, "report", slow_counting_report)

    results: list[tuple[int, bytes]] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(_get(f"{server}{_QC_QUERY}"))
        )
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert [status for status, _ in results] == [200, 200]
    assert results[0][1] == results[1][1]

    # A distinct cache key is not deduplicated against the first.
    status, _body = _get(f"{server}{_QC_QUERY.replace('row_mode=bar', 'row_mode=all')}")
    assert status == 200
    assert len(calls) == 2
