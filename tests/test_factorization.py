"""Independent differential gates for shared native plans and QC cohorts."""
import itertools

import numpy as np
import polars as pl
import pytest
from sklearn.neighbors import KDTree

from mill import discover, feature_registry, qc
from mill.expressions import Graph, evaluate
from mill.pairwise import Workspace, radius_counts
from mill.table import build_from_spec
from mill.views import ALL


def _raw(n=256):
    rng = np.random.default_rng(814)
    close = 100 + np.cumsum(rng.normal(size=n))
    values = {name: rng.uniform(1, 100, n) for name in feature_registry._RAW_COLUMNS}
    values.update(close=close, open=close + 0.1, high=close + 1, low=close - 1)
    return pl.DataFrame(values).with_columns(
        pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1) + pl.duration(hours=n-1),
                          interval="1h", time_zone="UTC", eager=True).alias("datetime"))


@pytest.mark.parametrize("view", ALL, ids=lambda view: view.NAME)
def test_view_graph_is_exactly_the_eager_reference(view):
    raw = _raw()
    expected = pl.concat([raw.select("datetime"), *[
        feature_registry.producer_for(item["factory"])(raw, **item.get("params", {}))
        for item in view.SPEC]], how="horizontal")
    actual = build_from_spec(raw, view.SPEC)
    assert actual.schema == expected.schema
    assert actual.equals(expected)


@pytest.mark.parametrize("kind", ["empty", "nullable", "chunked"])
def test_native_view_plans_preserve_empty_null_and_chunk_layout_behavior(kind):
    raw = _raw()
    if kind == "empty":
        raw = raw.head(0)
    elif kind == "nullable":
        raw = raw.with_columns(pl.when(pl.int_range(pl.len()) % 13 == 0).then(None)
                               .otherwise(pl.col(c)).alias(c) for c in raw.columns if c != "datetime")
    else:
        raw = pl.concat([raw[:73], raw[73:]], rechunk=False)
    for view in ALL:
        requests = [(feature_registry.producer_for(item["factory"]), item.get("params", {})) for item in view.SPEC]
        for actual, (producer, params) in zip(evaluate(raw, requests, errors=True), requests):
            try:
                expected = producer(raw, **params)
            except Exception as exc:
                assert isinstance(actual, type(exc)), view.NAME
            else:
                assert not isinstance(actual, Exception), view.NAME
                assert actual.schema == expected.schema and actual.equals(expected), view.NAME


def test_a_failing_native_factory_does_not_suppress_other_catalog_outputs():
    from mill.feature_registry import _limen_producer
    def invalid(raw):
        return raw.with_columns(pl.col("missing").alias("bad"))
    raw = _raw()
    good = feature_registry.producer_for("parkinson_volatility")
    result = evaluate(raw, [(good, {"window": 1}), (_limen_producer(invalid), {}), (good, {"window": 4})], errors=True)
    assert result[0].equals(good(raw, window=1))
    assert isinstance(result[1], pl.exceptions.ColumnNotFoundError)
    assert result[2].equals(good(raw, window=4))


@pytest.mark.parametrize("error", [RuntimeError, ZeroDivisionError])
def test_python_native_factory_failures_retain_catalog_skip_behavior(error, monkeypatch, tmp_path):
    from mill import pool_store, serve
    from mill.feature_registry import _limen_producer
    def bad(raw):
        raise error("factory failure")
    registry = {"one": {"producer": feature_registry._raw_producer("close"), "source": "raw", "params": ()},
                "bad": {"producer": _limen_producer(bad), "source": "raw", "params": ()},
                "two": {"producer": feature_registry._raw_producer("open"), "source": "raw", "params": ()}}
    monkeypatch.setattr(feature_registry, "REGISTRY", registry)
    monkeypatch.setenv("MILL_CACHE_DIR", str(tmp_path))
    raw = _raw()
    ordinary = discover.catalog_pool(raw)
    persisted, _, _ = pool_store.pool_for(raw, "1h_time_bars", None, serve._frame_digest(raw))
    assert ordinary.skipped == persisted.skipped == ("bad",)
    assert ordinary.names == persisted.names == ("close", "open")
    np.testing.assert_array_equal(ordinary.values, persisted.values)


def test_protocol_specific_runtime_error_falls_back_to_real_frame():
    from mill.feature_registry import _limen_producer
    def needs_frame(frame):
        if not isinstance(frame, pl.DataFrame):
            raise RuntimeError("requires an eager frame")
        return frame.with_columns((pl.col("close")*2).alias("double"))
    raw, producer = _raw(), _limen_producer(needs_frame)
    assert evaluate(raw, [(producer, {})], errors=True)[0].equals(producer(raw))


@pytest.mark.parametrize("swap", [False, True])
def test_raw_column_renames_and_swaps_preserve_original_lineage(swap):
    from mill.feature_registry import _limen_producer
    def renamed(frame):
        if swap:
            return frame.rename({"open": "close", "close": "open"}).with_columns(
                (pl.col("close")-pl.col("open")).alias("spread"))
        return frame.rename({"close": "price"}).with_columns((pl.col("price")*2).alias("double"))
    raw, producer = _raw(), _limen_producer(renamed)
    graph = Graph(raw)
    actual = graph.collect([producer(graph.frame())])[0]
    assert actual.equals(producer(raw))


def test_native_literal_round_trip_preserves_signed_zero():
    from mill.feature_registry import _limen_producer
    def signed(frame):
        return frame.with_columns(pl.lit(-0.).alias("negative"), pl.lit(0.).alias("positive"))
    raw, producer = _raw(), _limen_producer(signed)
    actual = evaluate(raw, [(producer, {})])[0]
    for column in actual.columns:
        np.testing.assert_array_equal(np.signbit(actual[column].to_numpy()),
                                      np.signbit(producer(raw)[column].to_numpy()))


def test_every_eligible_factory_matches_its_original_producer():
    raw = _raw()
    for name, entry in feature_registry.REGISTRY.items():
        if any(p["default"] is None for p in entry["params"]):
            continue
        producer = entry["producer"]
        try:
            expected = producer(raw)
        except Exception as error:
            with pytest.raises(type(error)):
                evaluate(raw, [(producer, {})])
        else:
            actual = evaluate(raw, [(producer, {})])[0]
            assert actual.schema == expected.schema, name
            assert actual.equals(expected), name


def test_variants_share_native_nodes_but_keep_parameter_column_names():
    graph = Graph(_raw())
    producer = feature_registry.producer_for("parkinson_volatility")
    first = producer(graph.frame(), window=1)
    before = len(graph.nodes)
    second = producer(graph.frame(), window=4)
    assert len(graph.nodes) == before + 1  # squared log range is shared
    actual = graph.collect([first, second])
    assert actual[0].columns == ["parkinson_volatility__window=1"]
    assert actual[1].columns == ["parkinson_volatility__window=4"]
    for part, window in zip(actual, (1, 4)):
        assert part.equals(producer(graph.raw, window=window))


def test_opaque_kernel_runs_once_on_its_original_path():
    from mill.feature_registry import _limen_producer
    calls = []
    def opaque(frame):
        def run(series):
            calls.append(1)
            return series * 2
        return frame.with_columns(pl.col("close").map_batches(run, return_dtype=pl.Float64).alias("opaque"))
    raw = _raw()
    result = evaluate(raw, [(_limen_producer(opaque), {})])[0]
    assert calls == [1]
    assert result["opaque"].equals(raw["close"] * 2, check_names=False)


def test_target_is_a_source_node_independent_of_feature_spec(monkeypatch):
    from mill import targets
    targets._RESULTS.clear()
    raw = _raw()
    original, calls = targets.apply, []
    def counted(frame, name):
        calls.append(name)
        return original(frame, name)
    monkeypatch.setattr(targets, "apply", counted)
    first = targets.for_source(raw, "next_return", ("source", 1))
    first.drop_in_place("next_return")
    again = targets.for_source(raw, "next_return", ("source", 1))
    assert again.equals(original(raw, "next_return").select("datetime", "next_return"))
    targets.for_source(raw, "next_return", ("source", 2))
    assert calls == ["next_return", "next_return"]


@pytest.mark.parametrize("n", [29, 30, 257, 5001, 20001])
def test_pair_factorization_is_bit_identical_with_pair_specific_samples(n):
    rng = np.random.default_rng(9821)
    arrays = {f"f{i}": rng.normal(size=n) for i in range(4)}
    arrays["f0"][:7] = np.nan
    arrays["f1"][::11] = np.nan
    arrays["f2"] = np.round(arrays["f2"])
    arrays["f3"][:] = 1
    work = Workspace(arrays, max_n=qc._PAIRWISE_MAX_N, mi_max_n=qc._PAIRWISE_MI_MAX_N,
                     min_n=qc._PAIRWISE_MIN_N)
    for left, right in itertools.combinations(arrays, 2):
        expected = qc._pair_metrics.__wrapped__(arrays[left], arrays[right])
        assert work.pair(left, right) == expected, (n, left, right)


def test_infinities_keep_spearman_and_reject_mi_like_reference():
    rng = np.random.default_rng(83)
    arrays = {"left": rng.normal(size=301), "right": rng.normal(size=301)}
    arrays["left"][::17] = np.inf
    work = Workspace(arrays, max_n=20000, mi_max_n=5000, min_n=30)
    assert work.pair("left", "right") == qc._pair_metrics.__wrapped__(*arrays.values())


@pytest.mark.parametrize("kind", ["continuous", "binary", "constant", "infinity"])
def test_shared_target_mi_matches_reference_for_all_target_kinds(kind):
    rng = np.random.default_rng(441)
    arrays = {str(i): rng.normal(size=517) for i in range(3)}
    arrays["0"][:13] = np.nan
    arrays["1"][::17] = np.nan
    y = rng.normal(size=517)
    if kind == "binary":
        y = (y > 0).astype(float)
    elif kind == "constant":
        y[:] = 1
    elif kind == "infinity":
        y[::11] = np.inf
    arrays["target"] = y
    blocks = {name: 24 for name in arrays}
    expected = {name: qc._edge_metrics.__wrapped__(arrays[name], y, 24, qc._seed_for(name))
                for name in ("0", "1", "2")}
    qc.clear_kernel_cache()
    actual = qc._edge_map(arrays, ("0", "1", "2"), y, blocks, "target")
    for name in expected:
        # The reference's decile spread is NaN for infinite targets. Preserve
        # that value too; ordinary tuple equality treats NaN as unequal.
        np.testing.assert_equal(actual[name], expected[name])


def test_sorted_radius_counts_match_kdtree_at_floating_point_boundaries():
    rng = np.random.default_rng(82)
    values = np.r_[rng.normal(size=180), np.zeros(8), np.nextafter(1., 0.), 1., np.nextafter(1., 2.)]
    radii = np.r_[np.abs(rng.normal(size=180)), np.zeros(8), 1., np.spacing(1.), np.spacing(1.)]
    expected = KDTree(values[:, None], metric="chebyshev").query_radius(
        values[:, None], radii, count_only=True)
    np.testing.assert_array_equal(radius_counts(values, np.sort(values), radii), expected)


def test_fused_bootstrap_matches_original_at_the_full_sampling_cap():
    rng = np.random.default_rng(88219)
    x, y = rng.normal(size=(2, qc._BOOTSTRAP_MAX_N))
    x = np.round(x, 1)
    idx = qc._bootstrap_block_indices(x.size, 24, 7, 2281)
    expected = [qc._pearson(qc._rankdata_1d(x[rows]), qc._rankdata_1d(y[rows])) for rows in idx]
    with qc._PARALLEL_KERNEL_LOCK:
        actual = qc._bootstrap_spearman_estimates(x, y, idx)
    np.testing.assert_array_equal(actual, expected)


def test_preparations_are_reused_only_for_identical_rows_and_remain_bounded(monkeypatch):
    rng = np.random.default_rng(882)
    arrays = {str(i): rng.normal(size=200) for i in range(8)}
    work = Workspace(arrays, max_n=20000, mi_max_n=5000, min_n=30)
    calls = []
    original = qc.stats.rankdata
    monkeypatch.setattr(qc.stats, "rankdata", lambda x: (calls.append(1), original(x))[1])
    for left, right in itertools.combinations(arrays, 2):
        work.pair(left, right)
    assert len(calls) == len(arrays)  # 8 rankings, rather than 2 * C(8,2)
    assert work.prepared.bytes <= work.prepared.budget
    monkeypatch.setenv("MILL_PAIR_WORK_MIB", "0")
    uncached = Workspace(arrays, max_n=20000, mi_max_n=5000, min_n=30)
    assert uncached.pair("0", "1") == work.pair("0", "1")
    assert len(uncached.prepared) == 0


def _original_null(y, ranks, shift):
    n = y.size
    indices = np.arange(shift, n-shift+1)
    best = np.zeros(indices.size)
    fy = np.conj(np.fft.rfft(y))[:, None]
    for c0 in range(0, ranks.shape[1], discover._FFT_CHUNK):
        block = ranks[:, c0:c0+discover._FFT_CHUNK]
        valid = ~np.isnan(block)
        v = valid.astype(float)
        s1 = np.fft.irfft(fy * np.fft.rfft(np.where(valid, block, 0.), axis=0), n=n, axis=0)[indices]
        n1 = np.rint(np.fft.irfft(fy * np.fft.rfft(v, axis=0), n=n, axis=0)[indices])
        with np.errstate(invalid="ignore", divide="ignore"):
            sep = np.abs((s1 - n1*(n1+1)/2)/(n1*(v.sum(axis=0)-n1)) - .5)
        sep[~np.isfinite(sep)] = 0
        best = np.maximum(best, sep.max(axis=1))
    return best


@pytest.mark.parametrize("unique", [False, True])
def test_shared_null_counts_match_reference_for_shared_and_unique_missingness(unique):
    rng = np.random.default_rng(941)
    values = rng.normal(size=(997, 35))
    for j in range(values.shape[1]):
        if unique:
            values[rng.random(997) < .03, j] = np.nan
        else:
            values[:j % 3, j] = np.nan
    ranks = discover.column_ranks(values)
    y = rng.integers(0, 2, size=997).astype(float)
    np.testing.assert_array_equal(discover.shift_null(y, ranks, 17), _original_null(y, ranks, 17))


def test_compact_ranks_are_lossless_and_keep_double_precision_statistics():
    rng = np.random.default_rng(8851)
    values = np.round(rng.normal(size=(10003, 37)), 1)
    values[rng.random(values.shape) < .03] = np.nan
    full = discover.column_ranks(values)
    compact = discover.column_ranks(values, compact=True)
    assert compact.dtype == np.float32 and compact.nbytes * 2 == full.nbytes
    np.testing.assert_array_equal(compact.astype(np.float64), full)
    y = rng.integers(0, 2, size=len(values)).astype(float)
    np.testing.assert_array_equal(discover._screen_auc(y, compact), discover.auc(y, full))
    np.testing.assert_array_equal(discover.shift_null(y, compact, 13, promote_ranks=True), discover.shift_null(y, full, 13))
    assert discover.rank_correlation(compact[:, 0], compact[:, 1]) == discover.rank_correlation(full[:, 0], full[:, 1])
    valid = np.isfinite(full[:, 0])
    assert discover.bootstrap_interval(compact[valid, 0], y[valid], 24, 77) == discover.bootstrap_interval(full[valid, 0], y[valid], 24, 77)
    assert discover._rank_dtype(2**23, True) == pl.Float32
    assert discover._rank_dtype(2**23+1, True) == pl.Float64
    assert discover._rank_dtype(10, False) == pl.Float64
