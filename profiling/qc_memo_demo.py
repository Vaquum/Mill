"""Show the kernel-memo win on the real per-request flows.

cold      : first QC of a k-feature set (cache empty) — unchanged from before.
warm      : identical re-run (every kernel hits) — the repeat-request case.
incremental: you had k-1 features cached and add one more — the workbench's
             common edit; only the new feature's kernels + its new pairs run.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import polars as pl

from mill import qc
from mill.get_data import GetData
from mill.table import build_from_spec

CACHE = Path.home() / ".cache" / "mill"
_FACTORIES = [
    "rsi", "mom", "roc", "rocp", "cci", "cmo", "willr", "atr", "natr", "trix",
    "apo", "ppo", "bop", "mfi", "kama", "tema",
]


def _raw() -> pl.DataFrame:
    m = sorted(glob.glob(str(CACHE / "1h_time_bars_*.parquet")),
               key=os.path.getsize, reverse=True)
    return GetData().from_parquet(m[0])


def _sub(raw, k):
    spec = [{"factory": f, "params": {}} for f in _FACTORIES[:k]]
    feats = build_from_spec(raw, spec)
    from mill import targets
    tdf = targets.apply(raw, "next_return").select("datetime", "next_return")
    return feats.join(tdf, on="datetime", how="left")


def _time(fn):
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000.0


def main():
    raw = _raw()
    k = 16
    sub_full = _sub(raw, k)
    sub_prev = _sub(raw, k - 1)

    # warm numba/sklearn/statsmodels once so timings reflect steady state.
    qc.report(sub_full, "next_return")

    qc.clear_kernel_cache()
    cold = _time(lambda: qc.report(sub_full, "next_return"))
    warm = _time(lambda: qc.report(sub_full, "next_return"))

    qc.clear_kernel_cache()
    qc.report(sub_prev, "next_return")          # cache the first k-1 features
    incr = _time(lambda: qc.report(sub_full, "next_return"))  # add the k-th

    print(f"rows={raw.height}  k={k}  target=next_return\n")
    print(f"cold  (cache empty)            : {cold:8.1f} ms")
    print(f"warm  (identical re-run)       : {warm:8.1f} ms   ({cold/warm:5.1f}x)")
    print(f"incremental (+1 feature on {k-1}) : {incr:8.1f} ms   ({cold/incr:5.1f}x)")


if __name__ == "__main__":
    main()
