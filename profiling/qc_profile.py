"""Profile the QC-through-the-API pipeline for one vs many features.

Drives the exact step sequence the /qc (and /data) endpoint runs in serve.py,
on real BTCUSDT bars, with the new Arrow load path:

    klines(bar)            # GetData.get_time_bars -> get_arrow_file (zero-copy)
    build_from_spec        # limen indicator/feature compute
    join target            # targets.apply + join
    _serialize             # /data JSON payload
    _group_for_qc          # row-mode grouping (default "bar" = passthrough)
    qc.report              # the statistical QC  <-- the focus
    grading.grade_map      # grade + verdicts
    json.dumps             # response body

It times every outer step and, inside qc.report, the four sections
(setup / health / pairwise / edge / regime) via monkeypatched timers, for a
sweep of feature counts k. Optionally runs cProfile for chosen k to attribute
time to the underlying library calls (adfuller / kpss / spearmanr / MI).

Run:  python profiling/qc_profile.py --ks 1,2,4,8,16 --reps 3
      python profiling/qc_profile.py --ks 1,16 --cprofile --reps 1
"""

from __future__ import annotations

import argparse
import cProfile
import glob
import io
import json
import os
import pstats
import statistics
import tempfile
import time
from pathlib import Path

import polars as pl

# Hermetic: arrow + cache live in a temp dir; never touch the host store.
_TMP = tempfile.mkdtemp(prefix="qc_profile_")
os.environ["MILL_ARROW_DIR"] = str(Path(_TMP) / "arrow")
os.environ["MILL_CACHE_DIR"] = str(Path(_TMP) / "cache")

from mill import grading, qc, targets  # noqa: E402
from mill.bars import arrow_series  # noqa: E402
from mill.settings import arrow_latest_path  # noqa: E402
from mill.table import build_from_spec, klines  # noqa: E402
from mill.feature_registry import producer_for, REGISTRY  # noqa: E402
from mill.serve import _date_filter, _group_for_qc, _serialize  # noqa: E402

CACHE = Path.home() / ".cache" / "mill"

# Candidate single-output indicators — each adds exactly one feature column, so
# we can dial the feature count k precisely.
_CANDIDATES = [
    "rsi", "mom", "roc", "rocp", "cci", "cmo", "willr", "atr", "natr", "trix",
    "apo", "ppo", "bop", "dx", "adx", "adxr", "mfi", "kama", "tema", "dema",
    "wma", "sma", "ema", "midpoint", "linearreg", "tsf", "minus_di", "plus_di",
    "ad", "obv", "trange", "avgprice", "medprice", "typprice", "wclprice",
]


def _seed_arrow_store(bar: str, parquet_glob: str) -> int:
    """Copy a real cached parquet into the temp Arrow store as the bar series."""
    matches = sorted(glob.glob(str(CACHE / parquet_glob)))
    if not matches:
        raise SystemExit(f"no cached parquet for {parquet_glob}")
    src = pl.read_parquet(matches[-1])
    values = [c for c in src.columns if c != "datetime"]
    framed = src.with_columns(
        pl.col("datetime").dt.cast_time_unit("ns").cast(pl.Int64).alias("ts")
    ).select("ts", *values)
    path = arrow_latest_path(arrow_series(bar))
    path.parent.mkdir(parents=True, exist_ok=True)
    framed.write_ipc(
        path, compression="uncompressed", record_batch_size=max(framed.height, 1)
    )
    return src.height


def _pick_single_col_factories(raw: pl.DataFrame, want: int) -> list[str]:
    chosen: list[str] = []
    for name in _CANDIDATES:
        if name not in REGISTRY:
            continue
        try:
            out = producer_for(name)(raw)
        except Exception:
            continue
        if out.width == 1 and out[out.columns[0]].cast(pl.Float64, strict=False).is_finite().any():
            chosen.append(name)
        if len(chosen) >= want:
            break
    if len(chosen) < want:
        raise SystemExit(f"only found {len(chosen)} single-col factories, need {want}")
    return chosen


def _median(fn, reps: int) -> tuple[object, float]:
    samples = []
    result = None
    for _ in range(reps):
        t = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t)
    return result, statistics.median(samples) * 1000.0  # ms


# ---- qc.report section timers (monkeypatched) -------------------------------
_SECTIONS = ("_health_rows", "_pairwise_rows", "_edge_rows", "_regime_rows")
_section_time: dict[str, float] = {}


def _install_section_timers() -> None:
    for name in _SECTIONS:
        orig = getattr(qc, name)

        def make(orig=orig, name=name):
            def wrapped(*a, **k):
                t = time.perf_counter()
                try:
                    return orig(*a, **k)
                finally:
                    _section_time[name] = _section_time.get(name, 0.0) + (
                        time.perf_counter() - t
                    )
            return wrapped

        setattr(qc, name, make())


def _report_with_sections(sub: pl.DataFrame, target: str | None):
    _section_time.clear()
    t = time.perf_counter()
    rep = qc.report(sub, target)
    total = (time.perf_counter() - t) * 1000.0
    sect = {k: v * 1000.0 for k, v in _section_time.items()}
    sect["setup(sort/cast/block)"] = total - sum(sect.values())
    return rep, total, sect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", default="1h_time_bars")
    ap.add_argument("--glob", default=None, help="cache parquet glob for the bar")
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--target", default="next_return")
    ap.add_argument("--row-mode", default="bar")
    ap.add_argument("--cprofile", action="store_true")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    bar = args.bar
    pglob = args.glob or f"{bar}_*.parquet"

    n_rows = _seed_arrow_store(bar, pglob)

    # Cold load = the new Arrow path (get_arrow_file) + GetData windowing/cache.
    t = time.perf_counter()
    raw = klines(bar)
    cold_ms = (time.perf_counter() - t) * 1000.0
    t = time.perf_counter()
    _ = klines(bar)  # warm: GetData parquet-cache hit
    warm_ms = (time.perf_counter() - t) * 1000.0

    target = args.target
    tdf = targets.apply(raw, target).select("datetime", target)

    factories = _pick_single_col_factories(raw, max(ks))
    _install_section_timers()

    # Warmup: compile numba, warm sklearn/statsmodels, at the largest k.
    warm_spec = [{"factory": f, "params": {}} for f in factories[: max(ks)]]
    warm_feat = build_from_spec(raw, warm_spec).join(tdf, on="datetime", how="left")
    qc.report(warm_feat, target)

    print(f"\n=== QC pipeline profile — bar={bar}  rows={raw.height} "
          f"target={target} row_mode={args.row_mode} reps={args.reps} ===")
    print(f"available single-col factories: {factories}")
    print(f"\ncold load (arrow get_arrow_file + window + cache): {cold_ms:8.1f} ms"
          f"   (one-time per bar/process)")
    print(f"warm load (GetData parquet-cache hit):            {warm_ms:8.1f} ms")
    print(f"in-process _RAW_CACHE hit in serve is ~0 (dict lookup)\n")

    hdr = (f"{'k':>3} | {'build':>8} {'tgtjoin':>8} {'serialize':>9} "
           f"{'group':>7} | {'qc.report':>9} {'grade':>7} {'json':>7} | "
           f"{'per-req':>8}")
    print(hdr)
    print("-" * len(hdr))

    section_rows = []
    cprofiles = []
    for k in ks:
        spec = [{"factory": f, "params": {}} for f in factories[:k]]
        feats, build_ms = _median(lambda: build_from_spec(raw, spec), args.reps)
        result, join_ms = _median(
            lambda: feats.join(tdf, on="datetime", how="left"), args.reps
        )
        _, ser_ms = _median(lambda: _serialize(result), args.reps)
        sub0 = _date_filter(result, "", "")
        grouped, group_ms = _median(
            lambda: _group_for_qc(sub0, args.row_mode, 5,
                                  target if target != "none" else None),
            args.reps,
        )
        reps_report = 1 if k >= 12 else max(1, args.reps - 1)
        report_total_samples = []
        rep = None
        sect = {}
        for _ in range(reps_report):
            rep, rtot, sect = _report_with_sections(
                grouped, target if target != "none" else None
            )
            report_total_samples.append(rtot)
        report_ms = statistics.median(report_total_samples)
        _, grade_ms = _median(
            lambda: grading.grade_map(rep, target=target if target != "none" else None),
            args.reps,
        )
        _, json_ms = _median(
            lambda: json.dumps(rep, allow_nan=False).encode(), args.reps
        )
        per_req = build_ms + join_ms + ser_ms + group_ms + report_ms + grade_ms + json_ms
        print(f"{k:>3} | {build_ms:8.1f} {join_ms:8.1f} {ser_ms:9.1f} "
              f"{group_ms:7.1f} | {report_ms:9.1f} {grade_ms:7.1f} {json_ms:7.1f} | "
              f"{per_req:8.1f}")
        section_rows.append((k, report_ms, sect))

        if args.cprofile:
            pr = cProfile.Profile()
            pr.enable()
            qc.report(grouped, target if target != "none" else None)
            pr.disable()
            buf = io.StringIO()
            st = pstats.Stats(pr, stream=buf).sort_stats("tottime")
            st.print_stats(12)
            cprofiles.append((k, buf.getvalue()))

    print("\n=== qc.report section breakdown (ms) ===")
    sh = f"{'k':>3} | {'setup':>10} {'health':>8} {'pairwise':>9} {'edge':>8} {'regime':>8} | {'total':>8}"
    print(sh)
    print("-" * len(sh))
    for k, rtot, sect in section_rows:
        print(f"{k:>3} | {sect.get('setup(sort/cast/block)',0):10.1f} "
              f"{sect.get('_health_rows',0):8.1f} {sect.get('_pairwise_rows',0):9.1f} "
              f"{sect.get('_edge_rows',0):8.1f} {sect.get('_regime_rows',0):8.1f} | {rtot:8.1f}")

    for k, text in cprofiles:
        print(f"\n=== cProfile qc.report  k={k}  (sorted by tottime, top 12) ===")
        # Trim pstats preamble noise for readability.
        for line in text.splitlines():
            if line.strip().startswith("ncalls") or (
                line.strip() and line.strip()[0].isdigit()
            ):
                print(line)


if __name__ == "__main__":
    main()
