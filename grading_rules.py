"""
QC grading thresholds and tables.

This file contains ONLY data — no logic. Edit values here to retune the
grading layer without touching its code.

All numeric ranges follow the spec convention: lower inclusive, upper
exclusive. Grade ladders are written top-to-bottom; the first row whose
threshold is satisfied wins. Grades are one of: "D", "C", "O", "S".
"""

from __future__ import annotations


# Grade ordering (lowest to highest). Used for min/max comparisons.
GRADE_RANK: dict[str, int] = {"D": 0, "C": 1, "O": 2, "S": 3}
GRADES_BY_RANK: tuple[str, ...] = ("D", "C", "O", "S")


# =============================================================================
# Threshold ladders.
# Each entry: (threshold, grade). Walk top-to-bottom; first row where
# `value >= threshold` wins. Final entry must have threshold = 0 to act as
# the catch-all.
# =============================================================================

MISSING_RATE: list[tuple[float, str]] = [
    (0.20, "D"),
    (0.05, "C"),
    (0.01, "O"),
    (0.0,  "S"),
]

ABS_SKEW: list[tuple[float, str]] = [
    (5.0, "D"),
    (2.0, "C"),
    (1.0, "O"),
    (0.0, "S"),
]

ABS_SPEARMAN_MAGNITUDE: list[tuple[float, str]] = [
    (0.15, "S"),
    (0.05, "O"),
    (0.02, "C"),
    (0.0,  "D"),
]

MUTUAL_INFORMATION: list[tuple[float, str]] = [
    (0.03,  "S"),
    (0.01,  "O"),
    (0.005, "C"),
    (0.0,   "D"),
]

ABS_DECILE_SPREAD: list[tuple[float, str]] = [
    (0.50, "S"),
    (0.20, "O"),
    (0.05, "C"),
    (0.0,  "D"),
]

ABS_MAX_PAIRWISE_SPEARMAN: list[tuple[float, str]] = [
    (0.95, "D"),
    (0.70, "C"),
    (0.30, "O"),
    (0.0,  "S"),
]

# Distance from 0.5 (bimodal: extremes good, mid-range bad).
MONOTONIC_DISTANCE: list[tuple[float, str]] = [
    (0.35, "S"),
    (0.20, "O"),
    (0.10, "C"),
    (0.0,  "D"),
]

# Per-test stationarity sub-grades.
ADF_P: list[tuple[float, str]] = [
    (0.20, "D"),
    (0.05, "C"),
    (0.01, "O"),
    (0.0,  "S"),
]

# KPSS uses opposite polarity: high p = stationary.
KPSS_P: list[tuple[float, str]] = [
    (0.10, "S"),
    (0.05, "O"),
    (0.01, "C"),
    (0.0,  "D"),
]

# regime std normalized by |spearman|.
REGIME_RELATIVE_STD: list[tuple[float, str]] = [
    (1.0, "D"),
    (0.5, "C"),
    (0.2, "O"),
    (0.0, "S"),
]


# =============================================================================
# Cardinality (discrete; depends on n_unique and n_rows).
# =============================================================================

CARDINALITY = {
    "constant_max": 1,         # n_unique <= 1 → D
    "binary": 2,               # n_unique == 2 → S
    "categorical_max": 50,     # 3..50 → S
    "broken_threshold_frac": 0.5,  # 50 < n_unique < n_rows*0.5 → C; >= → S
}


# =============================================================================
# Stationarity composite truth table.
# Indexed as STATIONARITY_TABLE[adf_grade][kpss_grade].
# =============================================================================

STATIONARITY_TABLE: dict[str, dict[str, str]] = {
    "D": {"D": "D", "C": "D", "O": "C", "S": "C"},
    "C": {"D": "D", "C": "C", "O": "C", "S": "O"},
    "O": {"D": "C", "C": "C", "O": "O", "S": "O"},
    "S": {"D": "C", "C": "O", "O": "O", "S": "S"},
}

# Special override: ADF=S AND KPSS=D → grade C with diagnostic note.
STATIONARITY_DIFF_TRENDING_NOTE = "difference-stationary, possibly trending"


# =============================================================================
# Spearman CI grading (composite of magnitude + CI).
# =============================================================================

# CI is off zero with margin smaller than this → C.
SPEARMAN_CI_NARROW_MIN_OFF_ZERO = 0.02
# Width threshold for "tight" CI (used in S path).
SPEARMAN_CI_TIGHT_WIDTH = 0.04
# Min off-zero margin for "S" path.
SPEARMAN_CI_STRONG_OFF_ZERO = 0.10


# =============================================================================
# Regime metric inputs.
# =============================================================================

# 4-chunk lookup (the canonical case from qc.py).
REGIME_SIGN_AGREEMENT_4: dict[str, str] = {
    "0/4": "D",
    "1/4": "D",
    "2/4": "C",
    "3/4": "O",
    "4/4": "S",
}

# Fallback for non-4-chunk: agreement_fraction = N_agree / N_regimes.
REGIME_AGREEMENT_FALLBACK: list[tuple[float, str]] = [
    (1.0,  "S"),  # exactly 1.0
    (0.51, "O"),  # > 0.5 but < 1.0  (use 0.51 floor; 0.5 itself maps to C)
    (0.5,  "C"),
    (0.0,  "D"),
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

# Edge STRONG threshold: at least N of 6 cells must be S, with no D or C.
EDGE_STRONG_S_COUNT = 3
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
