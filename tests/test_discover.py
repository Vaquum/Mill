"""Find features: the ranking, the chance check, the rule and the endpoint.

Oracles are independent of the code under test: a pairwise count for AUC, a
direct loop over np.roll for the FFT null, unit weights for the bootstrap AUC,
hand-written windows for the labels, and Limen's own validator for the manifest.
The endpoint runs on the synthetic hermetic Arrow store of conftest.
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from limen.yaml import validate

from mill import discover, serve
from mill.feature_registry import REGISTRY

from conftest import ArrowStore, ns

_AUTH = "Basic " + base64.b64encode(b"u:p").decode()
_H = 3_600_000


def _pairwise_auc(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x)
    pos, neg = x[valid & (y == 1)], x[valid & (y == 0)]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (pos.size * neg.size)


_X = np.array([
    [1.0, 9.0, 0.0],
    [2.0, 8.0, 0.0],
    [2.0, np.nan, 1.0],
    [3.0, 7.0, 1.0],
    [np.nan, 6.0, 0.0],
    [5.0, 5.0, 1.0],
    [4.0, 1.0, 0.0],
    [6.0, 2.0, 1.0],
])
_Y = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=float)


def test_auc_from_ranks_matches_a_pairwise_count() -> None:
    got = discover.auc(_Y, discover.column_ranks(_X))
    want = [_pairwise_auc(_X[:, c], _Y) for c in range(_X.shape[1])]
    assert np.allclose(got, want)


def test_fft_null_matches_a_direct_loop_over_shifts() -> None:
    ranks = discover.column_ranks(_X)
    min_shift = 2
    direct = [
        np.nanmax(np.abs(discover.auc(np.roll(_Y, s), ranks) - 0.5))
        for s in range(min_shift, _Y.size - min_shift + 1)
    ]
    assert np.allclose(discover.shift_null(_Y, ranks, min_shift), direct)


def test_weighted_auc_with_unit_weights_is_the_auc() -> None:
    x = _X[:, 0][np.isfinite(_X[:, 0])]
    y = _Y[np.isfinite(_X[:, 0])]
    got = discover.weighted_auc(x, y, np.ones((1, x.size)))
    assert np.isclose(got[0], _pairwise_auc(x, y))


def test_inside_marks_overlapping_bars_and_lead_marks_the_bars_before() -> None:
    ms = np.arange(10, dtype=np.int64) * _H
    window = [(int(4.5 * _H), int(6 * _H))]
    marked, left_out = discover.target(ms, _H, window, "inside", 2)
    assert np.flatnonzero(marked).tolist() == [4, 5]
    assert not left_out.any()
    marked, left_out = discover.target(ms, _H, window, "lead", 2)
    assert np.flatnonzero(marked).tolist() == [2, 3]
    assert np.flatnonzero(left_out).tolist() == [4, 5]


def test_rule_hits_follow_limen_persistence() -> None:
    values = np.array([0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, np.nan])
    assert discover.rule_hits(values, "x", ">", 1.0, 1).tolist() == [False, True, True, False, True, True, True, False]
    assert discover.rule_hits(values, "x", ">", 1.0, 2).tolist() == [False, False, True, False, False, True, True, False]


def test_threshold_maximizes_youden_between_observed_values() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    assert discover.threshold(x, y, ">") == 3.5
    assert discover.threshold(-x, y, "<") == -3.5


def test_threshold_keeps_the_split_it_selects() -> None:
    # Six significant digits would round this cut to 1.0 and flag neither row.
    x = np.array([1.000001, 1.000002])
    cut = discover.threshold(x, np.array([True, False]), "<")
    assert discover.rule_hits(x, "x", "<", cut, 1).tolist() == [True, False]
    # A column that cannot separate has no rule, not one that flags nothing.
    assert discover.threshold(np.ones(4), np.array([True, False, True, False]), ">") is None


_SOURCE = serve._BAR_DATA_SOURCES["1h_time_bars"]


def _run(values: np.ndarray, windows: list[tuple[int, int]], top: int = 5) -> dict:
    """discover() on a hand-made pool of hourly bars; windows are [start, stop) bar indices."""
    n, c = values.shape
    ms = discover.RESEARCH_START_MS + np.arange(n, dtype=np.int64) * _H
    names = tuple(f"c{i}" for i in range(c))
    pool = discover.Pool(names=names, natives=names, factories=("close",) * c, sources=("raw",) * c,
                         params=({},) * c, ms=ms, values=values, skipped=())
    span = (int(ms[0]), int(ms[-1]) + _H)
    return discover.discover(pool, step_ms=_H, windows=[(int(ms[a]), int(ms[b - 1]) + _H) for a, b in windows],
                             mode="inside", lead_bars=4, context=span, display=span, top=top, max_points=n,
                             bar_source=_SOURCE, label="test")


def test_beyond_chance_needs_an_adjusted_p_of_one_in_twenty() -> None:
    # A perfect separation with only two shifts to compare against cannot be beyond chance.
    out = _run(np.array([[1.0], [2.0], [3.0]]), [(2, 3)])
    assert out["chance"]["shifts"] == 2 and out["chance"]["critical"] is None
    assert out["candidates"][0]["separation"] == 0.5 and not out["candidates"][0]["beyond_chance"]
    # Among hundreds of shifts the same separation is, and the call agrees with p and the critical value.
    rng = np.random.default_rng(7)
    starts = np.sort(rng.choice(np.arange(10, 390), size=8, replace=False))
    windows = [(int(a), int(a) + 1) for a in starts]
    values = rng.normal(size=(400, 4))
    values[starts, 0] += 100.0
    out = _run(values, windows)
    leader = out["candidates"][0]
    assert leader["column"] == "c0" and leader["beyond_chance"] and leader["separation"] > out["chance"]["critical"]
    assert all(c["beyond_chance"] == (c["p_adjusted"] <= 0.05) for c in out["candidates"])


def test_no_column_defined_on_the_marked_bars_reports_no_best() -> None:
    values = np.arange(40, dtype=np.float64)[:, None].repeat(2, axis=1)
    values[10:14] = np.nan
    out = _run(values, [(10, 14)])
    assert out["screened"] == 0 and out["candidates"] == [] and out["chance"]["best"] is None


def test_intervals_need_thirty_events_where_the_column_is_defined() -> None:
    rng = np.random.default_rng(3)
    starts = np.sort(rng.choice(np.arange(5, 1995, 40), size=40, replace=False))
    values = rng.normal(size=(2000, 2))
    values[starts, 0] += 1.0
    # c1 is defined on one marked and one unmarked bar: a perfect but two-row separation.
    values[:, 1] = np.nan
    values[starts[0], 1], values[starts[0] + 1, 1] = 1.0, 0.0
    windows = [(int(a), int(a) + 1) for a in starts]
    out = _run(values, windows)
    assert out["intervals"] and [c["column"] for c in out["candidates"]] == ["c0", "c1"]
    assert out["candidates"][0]["ci90"] is not None and out["candidates"][1]["ci90"] is None
    # Asking for fewer leaders returns a prefix of the same order.
    assert [c["column"] for c in _run(values, windows, top=1)["candidates"]] == ["c0"]


def test_research_window_refuses_spans_outside_it() -> None:
    with pytest.raises(ValueError, match="research window"):
        discover.research_bounds(discover.parse_ms("2023-12-31T00:00:00Z"), discover.parse_ms("2024-01-02T00:00:00Z"))
    with pytest.raises(ValueError, match="research window"):
        discover.research_bounds(discover.parse_ms("2026-05-15T00:00:00Z"), discover.parse_ms("2026-05-16T00:00:01Z"))
    discover.research_bounds(discover.parse_ms("2026-05-15T00:00:00Z"), discover.parse_ms("2026-05-16T00:00:00Z"))


def test_rule_manifest_validates_with_limen() -> None:
    params = {p["name"]: p["default"] for p in REGISTRY["roc"]["params"]}
    rule = {"column": "roc_10", "operator": ">", "value": 1.25, "persistence_n": 2}
    source = serve._BAR_DATA_SOURCES["1h_time_bars"]
    for context in ((discover.parse_ms("2025-01-01"), discover.parse_ms("2025-07-01")),
                    (discover.parse_ms("2024-01-01"), discover.parse_ms("2026-05-16"))):
        manifest = discover.manifest(bar_source=source, factory="roc", source="limen.indicators", params=params,
                                     rule=rule, name="find_features_test", description="roc_10 > 1.25", context=context)
        result = validate(manifest)
        assert result.valid, [e.message for e in result.errors]
        splits = manifest["sfd"]["manifest"]["split_dates"]
        assert splits["train_start"] < splits["val_start"] < splits["test_start"] <= splits["test_end"]
    assert "not out of sample" in manifest["metadata"]["description"]


# Ten days from the research window's start, and its last ten days, so a default context reaches both ends.
_TS = ([ns(2024, 1, 1) + i * 3_600_000_000_000 for i in range(240)]
       + [ns(2026, 5, 6) + i * 3_600_000_000_000 for i in range(240)])


@pytest.fixture
def server(arrow_store: ArrowStore, monkeypatch: pytest.MonkeyPatch):
    arrow_store.write("time_1h", _TS)
    for name, value in (("_RESULTS", {}), ("_RAW_CACHE", {}), ("_BAR_LOCKS", {}), ("_POOLS", {}), ("_POOL_LOCKS", {})):
        monkeypatch.setattr(serve, name, value)
    monkeypatch.setattr(serve, "_VIEW_SPECS", {"ochlv": [{"factory": "close", "params": {}},
                                                         {"factory": "volume", "params": {}}]})
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


def _post(url: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(url + "/discover", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": _AUTH, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_discover_ranks_a_view_against_one_window(server: str) -> None:
    status, out = _post(server, {
        "bar": "1h_time_bars", "pool": {"kind": "view", "name": "ochlv"},
        "windows": [{"start": "2024-01-05T00:00:00Z", "end": "2024-01-05T12:00:00Z"}],
        "context": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-11T00:00:00Z"},
    })
    assert status == 200, out
    assert out["target"]["marked_bars"] == 12
    assert out["evidence"] == "description" and out["intervals"] is False
    assert out["chance"]["shifts"] == 240 - 2 * 12 + 1
    assert out["pool"] == {"kind": "view", "name": "ochlv", "columns": 2, "skipped": [], "cached": False,
                           "built_ms": out["pool"]["built_ms"]}
    # The fixture's columns are ramps over time, so both columns rank like time and form one cluster.
    assert len(out["candidates"]) == 1 and out["candidates"][0]["similar"]
    leader = out["candidates"][0]
    assert leader["rule"]["manifest"]["sfd"]["manifest"]["strategy"]["entry"] == "entry"
    assert len(leader["series"]["values"]) == 240


def test_discover_default_context_includes_the_last_research_day(server: str) -> None:
    status, out = _post(server, {"pool": {"kind": "view", "name": "ochlv"},
                                 "windows": [{"start": "2026-05-15T00:00:00Z", "end": "2026-05-16T00:00:00Z"}]})
    assert status == 200, out
    assert out["target"]["marked_bars"] == 24
    assert out["context"]["end_ms"] == discover.parse_ms("2026-05-16T00:00:00Z")


def test_discover_refuses_windows_outside_the_research_window(server: str) -> None:
    status, out = _post(server, {"windows": [{"start": "2023-06-01T00:00:00Z", "end": "2023-06-02T00:00:00Z"}]})
    assert status == 400 and out["error"] == "invalid_request"
    assert "research window" in out["detail"]
