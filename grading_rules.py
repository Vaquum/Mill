"""
QC grading thresholds and tables.

This file contains ONLY data — no logic. Edit values here to retune the
grading layer without touching its code.

All numeric ranges follow the spec convention: lower inclusive, upper
exclusive. Grade ladders are written top-to-bottom; the first row whose
threshold is satisfied wins. Grades are one of: "F", "D", "B", "A"
(no C in the scale — D is the middling grade, B is the next step up).
"""

from __future__ import annotations


# Grade ordering (lowest to highest). Used for min/max comparisons.
GRADE_RANK: dict[str, int] = {"F": 0, "D": 1, "B": 2, "A": 3}
GRADES_BY_RANK: tuple[str, ...] = ("F", "D", "B", "A")


# =============================================================================
# Threshold ladders.
# Each entry: (threshold, grade). Walk top-to-bottom; first row where
# `value >= threshold` wins. Final entry must have threshold = 0 to act as
# the catch-all.
# =============================================================================

MISSING_RATE: list[tuple[float, str]] = [
    (0.20, "F"),
    (0.05, "D"),
    (0.01, "B"),
    (0.0,  "A"),
]

ABS_SKEW: list[tuple[float, str]] = [
    (5.0, "F"),
    (2.0, "D"),
    (1.0, "B"),
    (0.0, "A"),
]

ABS_SPEARMAN_MAGNITUDE: list[tuple[float, str]] = [
    (0.15, "A"),
    (0.05, "B"),
    (0.02, "D"),
    (0.0,  "F"),
]

MUTUAL_INFORMATION: list[tuple[float, str]] = [
    (0.03,  "A"),
    (0.01,  "B"),
    (0.005, "D"),
    (0.0,   "F"),
]

ABS_DECILE_SPREAD: list[tuple[float, str]] = [
    (0.50, "A"),
    (0.20, "B"),
    (0.05, "D"),
    (0.0,  "F"),
]

ABS_MAX_PAIRWISE_SPEARMAN: list[tuple[float, str]] = [
    (0.95, "F"),
    (0.70, "D"),
    (0.30, "B"),
    (0.0,  "A"),
]

# Distance from 0.5 (bimodal: extremes good, mid-range bad).
MONOTONIC_DISTANCE: list[tuple[float, str]] = [
    (0.35, "A"),
    (0.20, "B"),
    (0.10, "D"),
    (0.0,  "F"),
]

# Per-test stationarity sub-grades.
ADF_P: list[tuple[float, str]] = [
    (0.20, "F"),
    (0.05, "D"),
    (0.01, "B"),
    (0.0,  "A"),
]

# KPSS uses opposite polarity: high p = stationary.
KPSS_P: list[tuple[float, str]] = [
    (0.10, "A"),
    (0.05, "B"),
    (0.01, "D"),
    (0.0,  "F"),
]

# regime std normalized by |spearman|.
REGIME_RELATIVE_STD: list[tuple[float, str]] = [
    (1.0, "F"),
    (0.5, "D"),
    (0.2, "B"),
    (0.0, "A"),
]


# =============================================================================
# Cardinality (discrete; depends on n_unique and n_rows).
# =============================================================================

CARDINALITY = {
    "constant_max": 1,         # n_unique <= 1 → F
    "binary": 2,               # n_unique == 2 → A
    "categorical_max": 50,     # 3..50 → A
    "broken_threshold_frac": 0.5,  # 50 < n_unique < n_rows*0.5 → D; >= → A
}


# =============================================================================
# Stationarity composite truth table.
# Indexed as STATIONARITY_TABLE[adf_grade][kpss_grade].
# =============================================================================

STATIONARITY_TABLE: dict[str, dict[str, str]] = {
    "F": {"F": "F", "D": "F", "B": "D", "A": "D"},
    "D": {"F": "F", "D": "D", "B": "D", "A": "B"},
    "B": {"F": "D", "D": "D", "B": "B", "A": "B"},
    "A": {"F": "D", "D": "B", "B": "B", "A": "A"},
}

# Special override: ADF=A AND KPSS=F → grade D with diagnostic note.
STATIONARITY_DIFF_TRENDING_NOTE = "difference-stationary, possibly trending"


# =============================================================================
# Spearman CI grading (composite of magnitude + CI).
# =============================================================================

# CI is off zero with margin smaller than this → D.
SPEARMAN_CI_NARROW_MIN_OFF_ZERO = 0.02
# Width threshold for "tight" CI (used in A path).
SPEARMAN_CI_TIGHT_WIDTH = 0.04
# Min off-zero margin for "A" path.
SPEARMAN_CI_STRONG_OFF_ZERO = 0.10


# =============================================================================
# Regime metric inputs.
# =============================================================================

# 4-chunk lookup (the canonical case from qc.py).
REGIME_SIGN_AGREEMENT_4: dict[str, str] = {
    "0/4": "F",
    "1/4": "F",
    "2/4": "D",
    "3/4": "B",
    "4/4": "A",
}

# Fallback for non-4-chunk: agreement_fraction = N_agree / N_regimes.
REGIME_AGREEMENT_FALLBACK: list[tuple[float, str]] = [
    (1.0,  "A"),  # exactly 1.0
    (0.51, "B"),  # > 0.5 but < 1.0  (use 0.51 floor; 0.5 itself maps to D)
    (0.5,  "D"),
    (0.0,  "F"),
]

# Below this |spearman|, regime_spearman_std is meaningless — return None.
REGIME_STD_MIN_EDGE = 0.05


# =============================================================================
# Composite verdict cell lists.
# =============================================================================

HEALTH_CELLS: tuple[str, ...] = (
    "missing_rate",
    "cardinality",
    "skew",
    "stationarity",
)

EDGE_CELLS: tuple[str, ...] = (
    "spearman",
    "mutual_information",
    "decile_spread",
    "monotonic_fraction",
    "regime_sign_agreement",
    "regime_spearman_std",
)

SELECTION_CELLS: tuple[str, ...] = ("redundancy",)

# Edge STRONG threshold: at least N of 6 cells must be A, with no F or D.
EDGE_STRONG_A_COUNT = 3
EDGE_TOTAL_CELLS = 6


# =============================================================================
# UI display mapping: which qc.report() metric rows should show each grade.
# A single grade may apply to multiple rows (e.g. stationarity informs both
# adf_p and kpss_p rows).
# =============================================================================

GRADE_TO_METRIC_ROWS: dict[str, list[str]] = {
    "missing_rate":          ["missing_rate"],
    "cardinality":           ["n_unique"],
    "skew":                  ["skew"],
    "stationarity":          ["adf_p", "kpss_p"],
    "spearman":              ["spearman"],
    "mutual_information":    ["mutual_information"],
    "decile_spread":         ["decile_spread"],
    "monotonic_fraction":    ["monotonic_fraction"],
    "regime_sign_agreement": ["regime_sign_agreement"],
    "regime_spearman_std":   ["regime_spearman_std"],
    "redundancy":            ["max_pairwise_spearman"],
}


# =============================================================================
# Reference: block-bootstrap settings used in qc.py (informational only).
# =============================================================================

BLOCK_SIZE_FALLBACK  = 24
BLOCK_SIZE_MAX       = 200
BLOCK_SIZE_THRESHOLD = 0.05
