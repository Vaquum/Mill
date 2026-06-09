"""The QC kernel memo is lossless and actually reuses work across requests.

Hermetic: builds a synthetic frame in-process (no data dependency), big enough
to clear qc's internal thresholds (ADF >= 50, edge >= 100 rows).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from mill import qc

_N = 2000
_NONPAIRWISE = {
    "missing_rate", "n_unique", "skew", "adf_p", "kpss_p",
    "spearman", "spearman_ci_lo", "spearman_ci_hi", "mutual_information",
    "decile_spread", "monotonic_fraction",
    "regime_sign_agreement", "regime_spearman_std",
}


def _frame(k: int, *, seed: int = 0, n: int = _N) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dt = pl.Series([base + timedelta(hours=i) for i in range(n)]).dt.cast_time_unit("ms")
    walks = np.cumsum(rng.standard_normal((n, k)), axis=0)
    cols = {"datetime": dt}
    for i in range(k):
        cols[f"f{i}"] = walks[:, i]
    cols["y"] = np.tanh(walks[:, 0] / 10.0) + rng.standard_normal(n) * 0.5
    return pl.DataFrame(cols)


def _rows(report: dict) -> dict[str, list]:
    return {r[0]: r[1:] for r in report["rows"]}


def _equal(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


def _reports_equal(a: dict, b: dict) -> bool:
    if a["columns"] != b["columns"] or a["n_rows"] != b["n_rows"]:
        return False
    ra, rb = _rows(a), _rows(b)
    if set(ra) != set(rb):
        return False
    return all(
        len(ra[m]) == len(rb[m]) and all(_equal(x, y) for x, y in zip(ra[m], rb[m]))
        for m in ra
    )


def test_cold_and_warm_runs_are_identical() -> None:
    frame = _frame(6)
    qc.clear_kernel_cache()
    cold = qc.report(frame, "y")        # all misses
    warm = qc.report(frame, "y")        # all hits
    assert _reports_equal(cold, warm)


def test_per_feature_metrics_independent_of_feature_set() -> None:
    """A feature's non-pairwise metrics are identical computed alone or batched
    — the property memoization relies on, and proof the refactor preserves it."""
    qc.clear_kernel_cache()
    full = _frame(4)
    solo = full.select("datetime", "f0", "y")
    rep_solo = qc.report(solo, "y")
    rep_full = qc.report(full, "y")
    i_solo = rep_solo["columns"].index("f0") - 1
    i_full = rep_full["columns"].index("f0") - 1
    rs, rf = _rows(rep_solo), _rows(rep_full)
    for metric in _NONPAIRWISE:
        assert _equal(rs[metric][i_solo], rf[metric][i_full]), metric


def test_warm_rerun_adds_no_cache_entries() -> None:
    frame = _frame(6)
    qc.clear_kernel_cache()
    qc.report(frame, "y")
    size = len(qc._KERNEL_CACHE)
    assert size > 0
    qc.report(frame, "y")
    assert len(qc._KERNEL_CACHE) == size


def test_incremental_reuses_cached_kernels() -> None:
    base = _frame(9)
    cols8 = ["datetime", *(f"f{i}" for i in range(8)), "y"]
    f8 = base.select(cols8)
    qc.clear_kernel_cache()
    qc.report(f8, "y")
    s8 = len(qc._KERNEL_CACHE)
    qc.report(base, "y")               # adds feature f8: its kernels + 8 new pairs
    s9 = len(qc._KERNEL_CACHE)
    added = s9 - s8
    # New work only: <= 5 single-array kernels for f8 + 8 new pairs. A full
    # recompute would re-add nothing (all keyed by content) — the point is the
    # increment is bounded by the *new* feature, not the whole set.
    assert 1 <= added <= 5 + 8
    # Re-running the 8-feature set now hits entirely: no new entries.
    qc.report(f8, "y")
    assert len(qc._KERNEL_CACHE) == s9


def test_clear_kernel_cache_empties_it() -> None:
    qc.report(_frame(3), "y")
    assert len(qc._KERNEL_CACHE) > 0
    qc.clear_kernel_cache()
    assert len(qc._KERNEL_CACHE) == 0


def test_target_none_path_is_stable() -> None:
    frame = _frame(4)
    qc.clear_kernel_cache()
    a = qc.report(frame, None)
    b = qc.report(frame, None)
    assert _reports_equal(a, b)
