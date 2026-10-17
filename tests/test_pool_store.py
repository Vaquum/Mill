import numpy as np
import polars as pl
import pytest

from mill import discover, feature_registry, pool_store, serve
from mill.get_data import GetData
from mill.memo import ByteLRU
from conftest import ns


def test_catalog_checkpoints_resume_and_mapped_pool_survives_ram_eviction(arrow_store, monkeypatch):
    arrow_store.write("time_1h", [ns(2024, 1, 1) + i * 3_600_000_000_000 for i in range(20)])
    raw = GetData().get_time_bars("1h_time_bars")
    calls = []
    def factory(name, values):
        def run(frame):
            calls.append(name)
            return pl.DataFrame({name: values * (frame.height // len(values))})
        return {"producer": run, "source": "raw", "params": ()}
    registry = {"a": factory("a", [1.0, None, np.inf, 2.0]),
                "b": factory("b", [2.0, 3.0, 4.0, 5.0])}
    monkeypatch.setattr(feature_registry, "REGISTRY", registry)
    digest = serve._frame_digest(raw)
    with pytest.raises(pool_store.PoolPending):
        pool_store.pool_for(raw, "1h_time_bars", None, digest, quantum=0)
    assert calls == ["a"]
    # A new invocation reconstructs progress from disk; no Python builder state.
    actual, cached, _ = pool_store.pool_for(raw, "1h_time_bars", None, digest, quantum=0)
    assert not cached and calls == ["a", "b"]
    assert isinstance(actual.values, np.memmap)
    monkeypatch.setattr(serve, "_POOLS", ByteLRU(1, lambda p: p.values.nbytes))
    monkeypatch.setattr(serve, "_PERSIST_POOLS", True)
    again, cached, _ = serve._pool_for("1h_time_bars", {"kind": "catalog"})
    assert cached and calls == ["a", "b"] and len(serve._POOLS) == 0
    expected = discover.catalog_pool(raw)
    assert actual.names == expected.names == again.names
    assert actual.params == expected.params
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(again.values, expected.values)


def test_empty_view_has_a_persisted_zero_column_pool(arrow_store):
    arrow_store.write("time_1h", [ns(2024, 1, 1)])
    raw = GetData().get_time_bars("1h_time_bars")
    pool, _, _ = pool_store.pool_for(raw, "1h_time_bars", [], serve._frame_digest(raw))
    assert pool.values.shape == (1, 0)
