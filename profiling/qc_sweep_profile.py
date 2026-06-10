"""Profile the QC sweep path: precompute once, assemble every subset report.

Builds a k-feature pool on real cached bars, then compares the two ways to QC
every feature combination of the requested sizes:

    per-subset qc.report   # what a sweep costs without primitives (warm memo)
    precompute + sweep     # kernels once, then pure aggregation per subset

Run:  python profiling/qc_sweep_profile.py --k 16 --sizes 1,2,3
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import tempfile
import time
from pathlib import Path

# Input parquet is read from the host cache (read-only); MILL_* dirs point at
# a temp dir so any Arrow/cache writes never touch the host store.
_TMP = tempfile.mkdtemp(prefix="qc_sweep_profile_")
os.environ["MILL_ARROW_DIR"] = str(Path(_TMP) / "arrow")
os.environ["MILL_CACHE_DIR"] = str(Path(_TMP) / "cache")

import polars as pl  # noqa: E402

from mill import qc, targets  # noqa: E402
from mill.feature_registry import REGISTRY, producer_for  # noqa: E402
from mill.get_data import GetData  # noqa: E402

CACHE = Path.home() / ".cache" / "mill"

_CANDIDATES = [
    "rsi", "mom", "roc", "rocp", "cci", "cmo", "willr", "atr", "natr", "trix",
    "apo", "ppo", "bop", "dx", "adx", "adxr", "mfi", "kama", "tema", "dema",
    "wma", "sma", "ema", "midpoint", "linearreg", "tsf", "minus_di", "plus_di",
    "ad", "obv", "trange", "avgprice", "medprice", "typprice", "wclprice",
]


def _pool(raw: pl.DataFrame, k: int, target: str) -> pl.DataFrame:
    df = raw.select("datetime")
    for name in _CANDIDATES:
        if name not in REGISTRY:
            continue
        try:
            out = producer_for(name)(raw)
        except Exception:
            continue
        if out.width == 1:
            df = df.hstack(out)
        if df.width - 1 >= k:
            break
    if df.width - 1 < k:
        raise SystemExit(f"only {df.width - 1} single-col factories, need {k}")
    tdf = targets.apply(raw, target).select("datetime", target)
    return df.join(tdf, on="datetime", how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar-glob", default="1h_time_bars_*.parquet")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sizes", default="1,2,3")
    ap.add_argument("--target", default="next_return")
    args = ap.parse_args()

    sizes = tuple(int(s) for s in args.sizes.split(","))
    matches = sorted(
        glob.glob(str(CACHE / args.bar_glob)), key=os.path.getsize, reverse=True
    )
    if not matches:
        raise SystemExit(f"no cached parquet for {args.bar_glob}")
    raw = GetData().from_parquet(matches[0])
    df = _pool(raw, args.k, args.target)
    n_combos = sum(math.comb(args.k, s) for s in sizes)
    print(f"rows={df.height}  pool k={args.k}  sizes={sizes}  combos={n_combos}")

    # Cold precompute: every kernel once over the pool.
    qc.clear_kernel_cache()
    t = time.perf_counter()
    prim = qc.precompute(df, args.target)
    pre_s = time.perf_counter() - t
    print(f"precompute (cold, all kernels):      {pre_s * 1000:10.1f} ms")

    # Sweep: assemble every subset report from the primitives.
    t = time.perf_counter()
    count = sum(1 for _ in qc.sweep(prim, sizes=sizes))
    sweep_s = time.perf_counter() - t
    print(f"sweep {count} subset reports:          {sweep_s * 1000:10.1f} ms"
          f"   ({sweep_s / count * 1e6:8.1f} us/subset)")

    # Reference: the same enumeration through qc.report (warm kernel memo).
    sample = list(qc.sweep(prim, sizes=(min(sizes),)))[:1]
    combo = sample[0][0]
    sel = ["datetime", *combo, args.target]
    t = time.perf_counter()
    qc.report(df.select(sel), args.target)
    one_report_s = time.perf_counter() - t
    print(f"one warm qc.report (k={len(combo)}):            {one_report_s * 1000:10.1f} ms"
          f"   -> x{n_combos} = {one_report_s * n_combos:6.1f} s for the same sweep")


if __name__ == "__main__":
    main()
