from __future__ import annotations

import polars as pl

from ..table import build_from_spec

NAME = "jolt"

SPEC = (
    {"factory": "rolling_volatility",   "params": {"column": "close", "window": 4}},
    {"factory": "rolling_volatility",   "params": {"column": "close", "window": 16}},
    {"factory": "realized_semivariance","params": {"window": 4}},
    {"factory": "realized_semivariance","params": {"window": 16}},
    {"factory": "parkinson_volatility", "params": {"window": 1}},
    {"factory": "parkinson_volatility", "params": {"window": 4}},
    {"factory": "jump_variation_proxy", "params": {"window": 16}},
    {"factory": "rolling_zscore",       "params": {"column": "liquidity_sum", "window": 96, "transform": "log1p"}},
    {"factory": "is_funding_hour",      "params": {}},
    {"factory": "is_us_open_hour",      "params": {}},
    {"factory": "calendar_time_features","params": {}},
)


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
