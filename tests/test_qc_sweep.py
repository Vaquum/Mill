"""precompute/report_for/sweep assemble subset reports that equal qc.report
run directly on the subset frame — the sweep is pure reuse, no new math.

Equality is plain ``==`` on the report dicts: report values are JSON-safe
(floats, ints, strings, None — never NaN), so dict equality is exact.

Hermetic: synthetic frame in-process, big enough to clear qc's internal
thresholds (ADF >= 50, edge >= 100 rows).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from mill import qc

_N = 800


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


def _direct(df: pl.DataFrame, combo: tuple[str, ...], target: str | None) -> dict:
    sel = ["datetime", *combo] + ([target] if target else [])
    return qc.report(df.select(sel), target)


def test_every_subset_report_matches_direct_report() -> None:
    df = _frame(5)
    qc.clear_kernel_cache()
    prim = qc.precompute(df, "y")
    for combo, rep in qc.sweep(prim, sizes=(1, 2, 3)):
        assert rep == _direct(df, combo, "y"), combo


def test_full_set_report_for_equals_report() -> None:
    df = _frame(4)
    qc.clear_kernel_cache()
    prim = qc.precompute(df, "y")
    assert qc.report_for(prim) == qc.report(df, "y")


def test_subset_without_target_matches_direct_report() -> None:
    df = _frame(4).drop("y")
    qc.clear_kernel_cache()
    prim = qc.precompute(df, None)
    assert prim.target is None
    combo = ("f1", "f3")
    assert qc.report_for(prim, combo) == _direct(df, combo, None)


def test_cols_are_normalized_to_pool_order() -> None:
    prim = qc.precompute(_frame(4), "y")
    assert qc.report_for(prim, ("f2", "f0")) == qc.report_for(prim, ("f0", "f2"))


def test_unknown_or_target_cols_raise() -> None:
    prim = qc.precompute(_frame(3), "y")
    with pytest.raises(ValueError, match="not in the precomputed pool"):
        qc.report_for(prim, ("f0", "nope"))
    with pytest.raises(ValueError, match="not in the precomputed pool"):
        qc.report_for(prim, ("y",))


def test_sweep_enumerates_all_combinations_in_pool_order() -> None:
    prim = qc.precompute(_frame(5), "y")
    combos = [combo for combo, _ in qc.sweep(prim, sizes=(1, 2, 3))]
    assert len(combos) == math.comb(5, 1) + math.comb(5, 2) + math.comb(5, 3)
    assert len(set(combos)) == len(combos)
    order = {c: i for i, c in enumerate(prim.feature_cols)}
    assert all(
        order[combo[i]] < order[combo[i + 1]]
        for combo in combos
        for i in range(len(combo) - 1)
    )


def test_sweep_runs_no_kernels_after_precompute() -> None:
    df = _frame(5)
    qc.clear_kernel_cache()
    prim = qc.precompute(df, "y")
    size = len(qc._KERNEL_CACHE)
    for _ in qc.sweep(prim, sizes=(1, 2, 3)):
        pass
    assert len(qc._KERNEL_CACHE) == size
