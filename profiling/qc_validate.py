"""Capture / compare exact QC outputs to prove an optimization is lossless.

Runs the real QC pipeline (build_from_spec -> target join -> _group_for_qc ->
qc.report -> grading.grade_map) on real cached BTCUSDT bars, over a deterministic
grid of (feature_count, target, row_mode). Serializes report + grades to a
canonical JSON (non-finite floats -> sentinels; float repr round-trips exactly),
so a byte-for-byte JSON match means bit-identical results.

    python profiling/qc_validate.py --dump  /tmp/qc_baseline.json   # before change
    python profiling/qc_validate.py --check /tmp/qc_baseline.json   # after change
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import polars as pl

from mill import grading, qc, targets
from mill.get_data import GetData
from mill.serve import _group_for_qc

CACHE = Path.home() / ".cache" / "mill"

# Fixed, deterministic single-output indicators (order matters, do not shuffle).
_FACTORIES = [
    "rsi", "mom", "roc", "rocp", "cci", "cmo", "willr", "atr", "natr", "trix",
    "apo", "ppo", "bop",
]
_KS = [1, 2, 3, 5, 8, 13]
_TARGETS = ["none", "next_return", "next_bar_up"]
_ROW_MODES = ["bar", "decile"]


def _load_raw(bar_glob: str) -> pl.DataFrame:
    matches = sorted(
        glob.glob(str(CACHE / bar_glob)), key=os.path.getsize, reverse=True
    )
    if not matches:
        raise SystemExit(f"no cached parquet for {bar_glob}")
    return GetData().from_parquet(matches[0])


def _canon(obj):
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__nan__"
        if math.isinf(obj):
            return "__inf__" if obj > 0 else "__-inf__"
        return obj
    return obj


def _build(raw: pl.DataFrame, k: int, target: str, row_mode: str):
    spec = [{"factory": f, "params": {}} for f in _FACTORIES[:k]]
    from mill.table import build_from_spec

    feats = build_from_spec(raw, spec)
    if target != "none":
        tdf = targets.apply(raw, target).select("datetime", target)
        feats = feats.join(tdf, on="datetime", how="left")
    sub = _group_for_qc(feats, row_mode, 5, target if target != "none" else None)
    return sub


def _run_all(raw: pl.DataFrame) -> dict:
    out: dict[str, object] = {}
    for target in _TARGETS:
        for row_mode in _ROW_MODES:
            for k in _KS:
                sub = _build(raw, k, target, row_mode)
                tgt = target if target != "none" else None
                rep = qc.report(sub, tgt)
                grades = grading.grade_map(rep, target=tgt)
                key = f"t={target}|rm={row_mode}|k={k}"
                out[key] = {"report": _canon(rep), "grades": _canon(grades)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--check")
    ap.add_argument("--bar-glob", default="1h_time_bars_*.parquet")
    args = ap.parse_args()

    raw = _load_raw(args.bar_glob)
    results = _run_all(raw)

    if args.dump:
        Path(args.dump).write_text(json.dumps(results, sort_keys=True, indent=0))
        print(f"dumped {len(results)} configs (rows={raw.height}) -> {args.dump}")
        return

    if args.check:
        baseline = json.loads(Path(args.check).read_text())
        now = json.loads(json.dumps(results, sort_keys=True))  # normalize tuples
        if set(baseline) != set(now):
            raise SystemExit(
                f"config set differs: {set(baseline) ^ set(now)}"
            )
        mismatches = 0
        for key in sorted(baseline):
            if baseline[key] != now[key]:
                mismatches += 1
                # Pinpoint the first differing metric row for honesty.
                b, n = baseline[key]["report"], now[key]["report"]
                if b.get("columns") == n.get("columns"):
                    brows = {r[0]: r[1:] for r in b["rows"]}
                    nrows = {r[0]: r[1:] for r in n["rows"]}
                    for m in brows:
                        if brows.get(m) != nrows.get(m):
                            print(f"DIFF {key} row={m}:")
                            print(f"  baseline: {brows.get(m)}")
                            print(f"  now:      {nrows.get(m)}")
                            break
                else:
                    print(f"DIFF {key}: columns differ")
        total = len(baseline)
        if mismatches:
            raise SystemExit(f"\nFAIL: {mismatches}/{total} configs differ")
        print(f"PASS: all {total} configs bit-identical (rows={raw.height})")
        return

    raise SystemExit("pass --dump PATH or --check PATH")


if __name__ == "__main__":
    main()
