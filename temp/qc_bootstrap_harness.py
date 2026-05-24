"""Temporary harness: profile and compare QC bootstrap implementations."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import polars as pl
from scipy import stats

from mill import qc
from mill.get_data import GetData
from mill.targets import apply as apply_target
from mill.views import gen4 as view

_BOOTSTRAP_ITER = qc._BOOTSTRAP_ITER
_BOOTSTRAP_MAX_N = qc._BOOTSTRAP_MAX_N


def _bootstrap_spearman_ci_ref(
    x: np.ndarray,
    y: np.ndarray,
    block: int,
    *,
    seed: int = 0,
    n_iter: int = _BOOTSTRAP_ITER,
    ci: float = 0.95,
) -> tuple[float | None, float | None]:
    n = x.size
    if n > _BOOTSTRAP_MAX_N:
        idx = np.linspace(0, n - 1, _BOOTSTRAP_MAX_N).astype(np.int64)
        x = x[idx]
        y = y[idx]
        n = x.size
    if n < block + 1:
        return None, None
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    n_blocks = max(1, (n + block - 1) // block)
    estimates = np.empty(n_iter)
    for i in range(n_iter):
        s = rng.choice(starts, size=n_blocks, replace=True)
        idx = (s[:, None] + np.arange(block)).ravel()[:n]
        v = stats.spearmanr(x[idx], y[idx]).statistic
        estimates[i] = v if not np.isnan(v) else 0.0
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(estimates, 100 * alpha)),
        float(np.percentile(estimates, 100 * (1 - alpha))),
    )


def _load_frame() -> pl.DataFrame:
    gd = GetData()
    raw = gd.get_time_bars(
        "1h", start_date_limit="2024-01-01", end_date_limit="2026-05-15"
    )
    raw_with_target = apply_target(raw, view.TARGET)
    df = view.build(raw)
    return df.join(
        raw_with_target.select(["datetime", view.TARGET]), on="datetime", how="left"
    )


def _paired_arrays(
    df: pl.DataFrame, feature: str, target: str
) -> tuple[np.ndarray, np.ndarray, int]:
    x = df[feature].cast(pl.Float64, strict=False).to_numpy()
    y = df[target].cast(pl.Float64, strict=False).to_numpy()
    mask = ~np.isnan(x) & ~np.isnan(y)
    block = qc._resolve_block(qc._block_size(x))
    return x[mask], y[mask], block


def _ci_close(
    ref: tuple[float | None, float | None],
    new: tuple[float | None, float | None],
    *,
    atol: float = 1e-15,
) -> bool:
    if ref == (None, None) and new == (None, None):
        return True
    if ref[0] is None or ref[1] is None or new[0] is None or new[1] is None:
        return False
    return bool(
        np.isclose(ref[0], new[0], rtol=0.0, atol=atol)
        and np.isclose(ref[1], new[1], rtol=0.0, atol=atol)
    )


def compare_bootstrap_pairs(
    df: pl.DataFrame,
    *,
    target: str,
    new_fn: Callable[..., tuple[float | None, float | None]],
) -> dict[str, object]:
    feature_cols = [c for c in df.columns if c not in {"datetime", target}]
    rows: list[dict[str, object]] = []
    for col in feature_cols:
        x, y, block = _paired_arrays(df, col, target)
        if x.size < 100:
            continue
        seed = qc._seed_for(col)
        ref = _bootstrap_spearman_ci_ref(x, y, block, seed=seed)
        new = new_fn(x, y, block, seed=seed)
        rows.append({
            "feature": col,
            "ref_lo": ref[0],
            "ref_hi": ref[1],
            "new_lo": new[0],
            "new_hi": new[1],
            "lo_diff": None if ref[0] is None or new[0] is None else abs(ref[0] - new[0]),
            "hi_diff": None if ref[1] is None or new[1] is None else abs(ref[1] - new[1]),
            "match": _ci_close(ref, new),
        })
    mismatches = [r for r in rows if not r["match"]]
    return {
        "pairs": rows,
        "n_compared": len(rows),
        "n_mismatch": len(mismatches),
        "max_lo_diff": max((r["lo_diff"] or 0.0) for r in rows) if rows else 0.0,
        "max_hi_diff": max((r["hi_diff"] or 0.0) for r in rows) if rows else 0.0,
        "passed": not mismatches,
    }


def compare_reports(
    df: pl.DataFrame,
    target: str,
    new_fn: Callable[..., tuple[float | None, float | None]],
) -> dict[str, object]:
    saved = qc._bootstrap_spearman_ci
    qc._bootstrap_spearman_ci = _bootstrap_spearman_ci_ref  # type: ignore[method-assign]
    ref = qc.report(df, target)
    qc._bootstrap_spearman_ci = new_fn  # type: ignore[method-assign]
    new = qc.report(df, target)
    qc._bootstrap_spearman_ci = saved  # type: ignore[method-assign]

    ci_rows = {"spearman_ci_lo", "spearman_ci_hi"}
    diffs: list[dict[str, object]] = []
    ref_map = {row[0]: row[1:] for row in ref["rows"]}
    new_map = {row[0]: row[1:] for row in new["rows"]}
    for metric in ci_rows:
        ref_vals = ref_map[metric]
        new_vals = new_map[metric]
        for col, ref_v, new_v in zip(ref["columns"][1:], ref_vals, new_vals, strict=True):
            if ref_v is None and new_v is None:
                continue
            if ref_v is None or new_v is None:
                diffs.append({
                    "metric": metric,
                    "column": col,
                    "ref": ref_v,
                    "new": new_v,
                })
                continue
            if not np.isclose(ref_v, new_v, rtol=0.0, atol=1e-15):
                diffs.append({
                    "metric": metric,
                    "column": col,
                    "ref": ref_v,
                    "new": new_v,
                })
    return {
        "ci_cell_diffs": diffs,
        "passed": not diffs,
    }


def profile_bootstrap(
    df: pl.DataFrame,
    *,
    target: str,
    new_fn: Callable[..., tuple[float | None, float | None]],
    repeats: int = 1,
) -> dict[str, float]:
    feature_cols = [c for c in df.columns if c not in {"datetime", target}]
    ref_total = 0.0
    new_total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        for col in feature_cols:
            x, y, block = _paired_arrays(df, col, target)
            if x.size < 100:
                continue
            _bootstrap_spearman_ci_ref(x, y, block, seed=qc._seed_for(col))
        ref_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        for col in feature_cols:
            x, y, block = _paired_arrays(df, col, target)
            if x.size < 100:
                continue
            new_fn(x, y, block, seed=qc._seed_for(col))
        new_total += time.perf_counter() - t0

    ref_avg = ref_total / repeats
    new_avg = new_total / repeats
    return {
        "ref_seconds": ref_avg,
        "new_seconds": new_avg,
        "speedup": ref_avg / new_avg if new_avg else float("inf"),
    }


def profile_report(
    df: pl.DataFrame,
    target: str,
    new_fn: Callable[..., tuple[float | None, float | None]],
    repeats: int = 1,
) -> dict[str, float]:
    saved = qc._bootstrap_spearman_ci
    ref_total = 0.0
    new_total = 0.0
    for _ in range(repeats):
        qc._bootstrap_spearman_ci = _bootstrap_spearman_ci_ref  # type: ignore[method-assign]
        t0 = time.perf_counter()
        qc.report(df, target)
        ref_total += time.perf_counter() - t0

        qc._bootstrap_spearman_ci = new_fn  # type: ignore[method-assign]
        t0 = time.perf_counter()
        qc.report(df, target)
        new_total += time.perf_counter() - t0

    qc._bootstrap_spearman_ci = saved  # type: ignore[method-assign]
    ref_avg = ref_total / repeats
    new_avg = new_total / repeats
    return {
        "ref_seconds": ref_avg,
        "new_seconds": new_avg,
        "speedup": ref_avg / new_avg if new_avg else float("inf"),
    }


def run(new_fn: Callable[..., tuple[float | None, float | None]]) -> None:
    df = _load_frame()
    target = view.TARGET
    qc._warmup_bootstrap()

    pair_cmp = compare_bootstrap_pairs(df, target=target, new_fn=new_fn)
    print("=== bootstrap pair equivalence ===")
    print(f"compared: {pair_cmp['n_compared']}, mismatches: {pair_cmp['n_mismatch']}")
    print(f"max lo diff: {pair_cmp['max_lo_diff']:.3e}")
    print(f"max hi diff: {pair_cmp['max_hi_diff']:.3e}")
    print(f"passed: {pair_cmp['passed']}")

    report_cmp = compare_reports(df, target, new_fn)
    print("\n=== full report CI equivalence ===")
    print(f"ci cell diffs: {len(report_cmp['ci_cell_diffs'])}")
    print(f"passed: {report_cmp['passed']}")

    boot_prof = profile_bootstrap(df, target=target, new_fn=new_fn)
    print("\n=== bootstrap profiling ===")
    print(f"ref:  {boot_prof['ref_seconds']:.2f}s")
    print(f"new:  {boot_prof['new_seconds']:.2f}s")
    print(f"speedup: {boot_prof['speedup']:.1f}x")

    report_prof = profile_report(df, target, new_fn)
    print("\n=== full report profiling ===")
    print(f"ref:  {report_prof['ref_seconds']:.2f}s")
    print(f"new:  {report_prof['new_seconds']:.2f}s")
    print(f"speedup: {report_prof['speedup']:.1f}x")


if __name__ == "__main__":
    run(qc._bootstrap_spearman_ci)
