from __future__ import annotations

import polars as pl

from bars import TIME_BAR_SECONDS
from table import build_from_spec

NAME = "nth_time_period"
BARS = tuple(TIME_BAR_SECONDS)

SPEC = (
    {"factory": "nth_minute",  "params": {}},
    {"factory": "nth_hour",    "params": {}},
    {"factory": "nth_day",     "params": {}},
    {"factory": "nth_month",   "params": {}},
    {"factory": "nth_quarter", "params": {}},
    {"factory": "nth_half",    "params": {}},
)


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
