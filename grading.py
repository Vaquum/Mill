"""
QC grading layer.

Takes a qc.report() output and produces per-feature grades + composite
verdicts (health / edge / selection). Pure-data input; no coupling to qc.py.

See grading_rules.py for the actual thresholds and tables.

Disclosures:

1. Cardinality needs n_unique and n_rows. Both are read directly from the
   qc.report() output (qc.py emits an "n_unique" row and a top-level
   "n_rows" field). Caller may still override n_rows via parameter.

2. Spearman CI grade: spec writes the O case as "CI off zero, OR (min(...)
   >= 0.05) → O" which is internally redundant. I treat O as the default
   "not-D, not-C, not-S" case; the explicit S path checks tightness AND
   off-zero ≥ 0.10. The 0.05 boundary in the spec is not used as a hard
   threshold in this implementation.

3. None handling in composite verdicts: spec is silent. I treat None as
   "skipped" — it never counts as D/C/O/S in any condition. health_verdict
   computes over non-None health cells; if all health cells are None,
   verdict is None. edge_verdict returns None when no edge cells have any
   grade (e.g., QC was run with target=none).

4. STRONG edge requires ≥ 3 of 6 cells to be graded S. None cells do not
   count as S. So a feature with 3 S grades and 3 None grades qualifies for
   STRONG, per the literal "≥ 3 of 6". This is the defensible reading of an
   ambiguous spec line.

5. Regime sign agreement fallback (non-4-chunk): the spec gives D / C /
   0.5–0.99 → O / 1.0 → S. The "0.5" boundary I treat as exactly 0.5 → C;
   any value strictly above 0.5 and below 1.0 → O; 1.0 → S; below 0.5 → D.

6. Stationarity special case (ADF=S AND KPSS=D) returns ("C", note). The
   note is preserved in the output dict under "stationarity_note" but is
   not part of the grade itself.

Each per-feature output dict additionally exposes:
   - "none_cells": list of metric names with None grade
   - "none_present": bool, True if any per-metric grade is None
so a downstream consumer can flag warnings on incomplete grading.
"""

from __future__ import annotations

import math
import re
from typing import Any

import grading_rules as R


# ----- helpers ---------------------------------------------------------------


def _finite(v: Any) -> bool:
    if v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _apply_ladder(value: Any, ladder: list[tuple[float, str]]) -> str | None:
    if not _finite(value):
        return None
    f = float(value)
    for threshold, grade in ladder:
        if f >= threshold:
            return grade
    return None


def _grade_min(*grades: str | None) -> str | None:
    valid = [g for g in grades if g is not None]
    if not valid:
        return None
    return min(valid, key=lambda g: R.GRADE_RANK[g])


# ----- per-metric grading ----------------------------------------------------


def grade_missing_rate(rate: Any) -> str | None:
    return _apply_ladder(rate, R.MISSING_RATE)


def grade_cardinality(n_unique: int | None, n_rows: int | None) -> str | None:
    if n_unique is None or n_rows is None or n_rows <= 0:
        return None
    if n_unique <= R.CARDINALITY["constant_max"]:
        return "D"
    if n_unique == R.CARDINALITY["binary"]:
        return "S"
    if n_unique <= R.CARDINALITY["categorical_max"]:
        return "S"
    if n_unique < n_rows * R.CARDINALITY["broken_threshold_frac"]:
        return "C"
    return "S"


def grade_skew(skew: Any) -> str | None:
    if not _finite(skew):
        return None
    return _apply_ladder(abs(float(skew)), R.ABS_SKEW)


def grade_stationarity(
    adf_p: Any, kpss_p: Any
) -> tuple[str | None, str | None]:
    if not _finite(adf_p) or not _finite(kpss_p):
        return None, None
    adf_grade = _apply_ladder(adf_p, R.ADF_P)
    kpss_grade = _apply_ladder(kpss_p, R.KPSS_P)
    if adf_grade is None or kpss_grade is None:
        return None, None
    if adf_grade == "S" and kpss_grade == "D":
        return "C", R.STATIONARITY_DIFF_TRENDING_NOTE
    combined = R.STATIONARITY_TABLE[adf_grade][kpss_grade]
    return combined, None


def _grade_spearman_ci(ci_low: Any, ci_high: Any) -> str | None:
    if not (_finite(ci_low) and _finite(ci_high)):
        return None
    lo = float(ci_low)
    hi = float(ci_high)
    if lo > hi:
        lo, hi = hi, lo
    if lo * hi <= 0.0:
        return "D"
    min_off = min(abs(lo), abs(hi))
    width = abs(hi - lo)
    if min_off < R.SPEARMAN_CI_NARROW_MIN_OFF_ZERO:
        return "C"
    if (
        min_off >= R.SPEARMAN_CI_STRONG_OFF_ZERO
        and width < R.SPEARMAN_CI_TIGHT_WIDTH
    ):
        return "S"
    return "O"


def grade_spearman(spearman: Any, ci_low: Any, ci_high: Any) -> str | None:
    if not _finite(spearman):
        return None
    mag_grade = _apply_ladder(abs(float(spearman)), R.ABS_SPEARMAN_MAGNITUDE)
    ci_grade = _grade_spearman_ci(ci_low, ci_high)
    if mag_grade is None and ci_grade is None:
        return None
    return _grade_min(mag_grade, ci_grade)


def grade_mutual_information(mi: Any) -> str | None:
    if not _finite(mi):
        return None
    return _apply_ladder(float(mi), R.MUTUAL_INFORMATION)


def grade_decile_spread(spread: Any) -> str | None:
    if spread is None:
        return None
    if not _finite(spread):
        return None
    return _apply_ladder(abs(float(spread)), R.ABS_DECILE_SPREAD)


def grade_monotonic_fraction(frac: Any) -> str | None:
    if not _finite(frac):
        return None
    distance = abs(float(frac) - 0.5)
    return _apply_ladder(distance, R.MONOTONIC_DISTANCE)


def grade_regime_sign_agreement(agreement: Any) -> str | None:
    if agreement is None or not isinstance(agreement, str):
        return None
    if agreement in R.REGIME_SIGN_AGREEMENT_4:
        return R.REGIME_SIGN_AGREEMENT_4[agreement]
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", agreement)
    if not m:
        return None
    n_agree = int(m.group(1))
    n_regimes = int(m.group(2))
    if n_regimes <= 0:
        return None
    frac = n_agree / n_regimes
    if frac >= 1.0:
        return "S"
    if frac > 0.5:
        return "O"
    if frac == 0.5:
        return "C"
    return "D"


def grade_regime_spearman_std(std: Any, spearman: Any) -> str | None:
    if not _finite(std) or not _finite(spearman):
        return None
    if abs(float(spearman)) < R.REGIME_STD_MIN_EDGE:
        return None
    relative = float(std) / abs(float(spearman))
    return _apply_ladder(relative, R.REGIME_RELATIVE_STD)


def grade_redundancy(max_pairwise_spearman: Any) -> str | None:
    if not _finite(max_pairwise_spearman):
        return None
    return _apply_ladder(
        abs(float(max_pairwise_spearman)), R.ABS_MAX_PAIRWISE_SPEARMAN
    )


# ----- composite verdicts ---------------------------------------------------


def health_verdict(grades: dict[str, str | None]) -> str | None:
    cells = [grades.get(c) for c in R.HEALTH_CELLS]
    valid = [g for g in cells if g is not None]
    if not valid:
        return None
    if any(g == "D" for g in valid):
        return "INVALID"
    if any(g == "C" for g in valid):
        return "WEAK"
    if all(g == "S" for g in valid):
        return "CLEAN"
    return "OK"


def edge_verdict(
    grades: dict[str, str | None], health: str | None
) -> str | None:
    if health == "INVALID":
        return None
    cells = {c: grades.get(c) for c in R.EDGE_CELLS}
    if all(g is None for g in cells.values()):
        return None
    sp = cells["spearman"]
    mi = cells["mutual_information"]
    if sp == "D" and mi == "D":
        return "NONE"
    if any(g == "D" for g in cells.values()):
        return "WEAK"
    if any(g == "C" for g in cells.values()):
        return "MODERATE"
    s_count = sum(1 for g in cells.values() if g == "S")
    if s_count >= R.EDGE_STRONG_S_COUNT:
        return "STRONG"
    return "USABLE"


def selection_verdict(grades: dict[str, str | None]) -> str | None:
    g = grades.get("redundancy")
    return {
        "D": "DUPLICATE",
        "C": "OVERLAPPING",
        "O": "MOSTLY_UNIQUE",
        "S": "UNIQUE",
    }.get(g)


def composite_label(
    health: str | None, edge: str | None, selection: str | None
) -> str:
    return "/".join(part or "-" for part in (health, edge, selection))


def grade_map(
    qc_report: dict[str, Any],
    n_rows: int | None = None,
    target: str | None = None,
) -> dict[str, dict[str, str]]:
    """
    Build a {metric_row: {feature: grade}} mapping for UI display.

    Re-keys grades by the qc.report() row name they should appear on
    (per grading_rules.GRADE_TO_METRIC_ROWS). One grade can land on
    multiple rows — e.g. stationarity is shown on both adf_p and kpss_p.
    """
    graded = grade_report(qc_report, n_rows=n_rows, target=target)
    out: dict[str, dict[str, str]] = {}
    for g in graded:
        fname = g["feature_name"]
        for key, val in g.items():
            if not key.startswith("grade_") or val is None:
                continue
            grade_name = key[len("grade_"):]
            for row_name in R.GRADE_TO_METRIC_ROWS.get(grade_name, []):
                out.setdefault(row_name, {})[fname] = val
    return out


# ----- main entry ------------------------------------------------------------


def grade_report(
    qc_report: dict[str, Any],
    n_rows: int | None = None,
    target: str | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a qc.report() output into per-feature grades.

    Parameters
    ----------
    qc_report:
        Dict as produced by qc.report(). Expects "columns", "rows", and
        (since the n_unique change in qc.py) an "n_rows" top-level field
        plus an "n_unique" row.
    n_rows:
        Optional override. If omitted, qc_report["n_rows"] is used.
        Falls back to 0 if neither is available, which makes cardinality
        grading return None.
    target:
        Optional target column name. Excluded from the per-feature output.

    Returns
    -------
    A list of per-feature dicts, in the same column order as the qc.report
    columns. Each dict carries the raw metric values plus the per-metric
    grades, the three composite verdicts, a list of skipped (None) cells,
    and a `none_present` flag.
    """
    columns = qc_report.get("columns") or []
    rows = qc_report.get("rows") or []
    if not columns or not rows:
        return []

    if n_rows is None:
        n_rows = int(qc_report.get("n_rows") or 0)

    metrics = {row[0]: row[1:] for row in rows}
    feature_cols = [c for c in columns[1:] if c != target]
    col_index = {c: i for i, c in enumerate(columns[1:])}

    out: list[dict[str, Any]] = []
    for feature in feature_cols:
        idx = col_index[feature]

        def get(metric_name: str, _idx: int = idx) -> Any:
            vals = metrics.get(metric_name)
            if vals is None or _idx >= len(vals):
                return None
            return vals[_idx]

        unique_fraction = get("unique_fraction")
        n_unique = get("n_unique")
        if not isinstance(n_unique, int):
            try:
                n_unique = int(n_unique) if n_unique is not None else None
            except (TypeError, ValueError):
                n_unique = None

        raw = {
            "feature_name": feature,
            "missing_rate": get("missing_rate"),
            "unique_fraction": unique_fraction,
            "n_unique": n_unique,
            "skew": get("skew"),
            "min": get("min"),
            "max": get("max"),
            "median": get("median"),
            "p01": get("p01"),
            "p99": get("p99"),
            "adf_p": get("adf_p"),
            "kpss_p": get("kpss_p"),
            "acf_lag_1": get("acf_lag_1"),
            "acf_lag_5": get("acf_lag_5"),
            "acf_lag_24": get("acf_lag_24"),
            "acf_lag_168": get("acf_lag_168"),
            "block_size": get("block_size"),
            "spearman": get("spearman"),
            "spearman_ci_lo": get("spearman_ci_lo"),
            "spearman_ci_hi": get("spearman_ci_hi"),
            "mutual_information": get("mutual_information"),
            "decile_spread": get("decile_spread"),
            "monotonic_fraction": get("monotonic_fraction"),
            "regime_sign_agreement": get("regime_sign_agreement"),
            "regime_spearman_std": get("regime_spearman_std"),
            "max_pairwise_spearman": get("max_pairwise_spearman"),
            "mean_pairwise_spearman": get("mean_pairwise_spearman"),
            "max_pairwise_mi": get("max_pairwise_mi"),
        }

        stationarity_grade, stationarity_note = grade_stationarity(
            raw["adf_p"], raw["kpss_p"]
        )

        grades: dict[str, str | None] = {
            "missing_rate":         grade_missing_rate(raw["missing_rate"]),
            "cardinality":          grade_cardinality(n_unique, n_rows),
            "skew":                 grade_skew(raw["skew"]),
            "stationarity":         stationarity_grade,
            "spearman":             grade_spearman(
                raw["spearman"], raw["spearman_ci_lo"], raw["spearman_ci_hi"]
            ),
            "mutual_information":   grade_mutual_information(
                raw["mutual_information"]
            ),
            "decile_spread":        grade_decile_spread(raw["decile_spread"]),
            "monotonic_fraction":   grade_monotonic_fraction(
                raw["monotonic_fraction"]
            ),
            "regime_sign_agreement": grade_regime_sign_agreement(
                raw["regime_sign_agreement"]
            ),
            "regime_spearman_std":  grade_regime_spearman_std(
                raw["regime_spearman_std"], raw["spearman"]
            ),
            "redundancy":           grade_redundancy(
                raw["max_pairwise_spearman"]
            ),
        }

        h = health_verdict(grades)
        e = edge_verdict(grades, h)
        s = selection_verdict(grades)

        graded = dict(raw)
        for k, g in grades.items():
            graded[f"grade_{k}"] = g
        none_cells = [k for k, g in grades.items() if g is None]
        graded["none_cells"] = none_cells
        graded["none_present"] = bool(none_cells)
        graded["stationarity_note"] = stationarity_note
        graded["health_verdict"] = h
        graded["edge_verdict"] = e
        graded["selection_verdict"] = s
        graded["composite_label"] = composite_label(h, e, s)

        out.append(graded)

    return out
