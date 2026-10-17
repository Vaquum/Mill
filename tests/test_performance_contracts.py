"""Numerical, freshness and resource contracts for engine optimizations."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import polars as pl
import pytest

from mill import discover, qc, serve, table
from mill.get_data import GetData, _DATA
from mill.memo import ByteLRU, Flights
from conftest import ns


def test_lru_reclaims_bytes_gradually_and_rejects_oversized_entries():
    cache = ByteLRU(6, len, max_entries=3)
    cache["a"], cache["b"] = b"aaa", b"bbb"
    assert cache["a"] == b"aaa"
    cache["c"] = b"ccc"
    assert list(cache) == ["a", "c"] and cache.bytes == 6
    cache["c"] = b"ccccccc"
    assert list(cache) == ["a"] and cache.bytes == 3
    assert cache.pop("a") == b"aaa" and cache.bytes == 0


def test_flight_keys_are_released_after_success_and_failure():
    flights = Flights()
    for key in range(100):
        with flights.hold(key):
            assert key in flights._items
    with pytest.raises(ValueError), flights.hold("failed"):
        raise ValueError()
    assert not flights._items


def test_replaced_source_with_same_timestamp_invalidates_memory_and_disk(arrow_store):
    arrow_store.write("time_1h", [ns(2024, 1, 1)])
    source = arrow_store.root / "time_1h" / "latest.arrow"
    old = GetData().get_time_bars("1h_time_bars")
    stamp = source.stat().st_mtime_ns
    replacement = source.with_name("replacement.arrow")
    frame = pl.read_ipc(source, memory_map=False).with_columns(pl.lit(987.0).alias("close"))
    frame.write_ipc(replacement)
    os.utime(replacement, ns=(stamp, stamp))
    replacement.replace(source)
    changed = GetData().get_time_bars("1h_time_bars")
    assert not changed.equals(old) and changed["close"][0] == 987.0
    _DATA.clear()
    assert GetData().get_time_bars("1h_time_bars").equals(changed)
    source.unlink()
    _DATA.clear()
    assert GetData().get_time_bars("1h_time_bars").equals(changed)


def test_getdata_returns_independent_dataframe_wrappers(arrow_store):
    arrow_store.write("time_1h", [ns(2024, 1, 1)])
    first = GetData().get_time_bars("1h_time_bars")
    first.drop_in_place("close")
    assert "close" in GetData().get_time_bars("1h_time_bars").columns


def test_incremental_specs_reuse_producers_but_isolate_source_versions(monkeypatch):
    calls = []
    raw = pl.DataFrame({"datetime": [1, 2]})
    def producer(factory):
        def run(frame, **params):
            calls.append((factory, params))
            return pl.DataFrame({factory: [params.get("n", 1)] * frame.height})
        return run
    monkeypatch.setattr(table, "producer_for", producer)
    monkeypatch.setattr(table, "_PRODUCERS", ByteLRU(1024, lambda f: f.estimated_size()))
    a, b = {"factory": "a"}, {"factory": "b", "params": {"n": 3}}
    table.build_from_spec(raw, [a], cache_key=("source", 1))
    both = table.build_from_spec(raw, [a, b], cache_key=("source", 1))
    assert len(calls) == 2 and both["b"].to_list() == [3, 3]
    table.build_from_spec(raw, [a], cache_key=("source", 2))
    assert len(calls) == 3


def test_concurrent_first_raw_load_is_single_flight(arrow_store, monkeypatch):
    arrow_store.write("time_1h", [ns(2024, 1, 1)])
    monkeypatch.setattr(serve, "_RAW_CACHE", {})
    calls = []
    real = serve.klines
    def counted(bar):
        calls.append(1)
        return real(bar)
    monkeypatch.setattr(serve, "klines", counted)
    with ThreadPoolExecutor(4) as executor:
        frames = list(executor.map(serve._raw_for, ["1h_time_bars"] * 4))
    assert len(calls) == 1 and all(f[0].equals(frames[0][0]) for f in frames)


def test_qc_frame_does_not_serialize_until_data_is_requested(arrow_store, monkeypatch):
    arrow_store.write("time_1h", [ns(2024, 1, 1)])
    monkeypatch.setattr(serve, "_RESULTS", {})
    monkeypatch.setattr(serve, "_RAW_CACHE", {})
    real = serve._serialize
    calls = []
    def counted(frame):
        calls.append(1)
        return real(frame)
    monkeypatch.setattr(serve, "_serialize", counted)
    payload, frame = serve._ensure("1h_time_bars", "none", [], serialize=False)
    assert payload is None and calls == []
    payload, again = serve._ensure("1h_time_bars", "none", [])
    assert again is frame and len(calls) == 1 and json.loads(payload)["rows"]


@pytest.mark.parametrize("kind", ["continuous", "ties", "flat", "infinity", "nan"])
def test_bootstrap_counted_ranks_are_bit_identical_to_sorting_each_sample(kind):
    rng = np.random.default_rng(982)
    x, y = rng.normal(size=(2, 257))
    if kind == "ties":
        x, y = np.round(x), np.round(y)
    elif kind == "flat":
        x[:] = 1
    elif kind == "infinity":
        x[::7], y[::11] = np.inf, -np.inf
    elif kind == "nan":
        x[0] = np.nan
    idx = qc._bootstrap_block_indices(x.size, 19, 12, 881)
    expected = []
    for rows in idx:
        value = qc._pearson(qc._rankdata_1d(x[rows]), qc._rankdata_1d(y[rows]))
        expected.append(0 if np.isnan(value) else value)
    with qc._PARALLEL_KERNEL_LOCK:
        actual = qc._bootstrap_spearman_estimates(x, y, idx)
    np.testing.assert_array_equal(actual, expected)


def test_discovery_bootstrap_batches_preserve_rng_draws_and_reductions():
    rng = np.random.default_rng(31)
    x, y = np.round(rng.normal(size=301)), rng.integers(0, 2, size=301).astype(bool)
    block, seed = 23, 41
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, x.size, size=(discover._BOOTSTRAP_DRAWS, (x.size + block - 1) // block))
    rows = ((starts[:, :, None] + np.arange(block)) % x.size).reshape(starts.shape[0], -1)[:, :x.size]
    flat = rows + (np.arange(discover._BOOTSTRAP_DRAWS) * x.size)[:, None]
    weights = np.bincount(flat.ravel(), minlength=discover._BOOTSTRAP_DRAWS * x.size).reshape(discover._BOOTSTRAP_DRAWS, x.size).astype(float)
    draws = discover.weighted_auc(x, y, weights)
    expected = tuple(np.quantile(draws[np.isfinite(draws)], [0.05, 0.95]))
    assert discover.bootstrap_interval(x, y, block, seed) == expected
