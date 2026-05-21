from __future__ import annotations

import polars as pl

from table import build_from_spec

NAME = 'gen3'

TARGET = 'abs_forward_vol_normalized_return'

SPEC = (
    {"factory": 'parkinson_volatility', "params": {'window': 1}},
    {"factory": 'parkinson_volatility', "params": {'window': 4}},
    {"factory": 'rolling_zscore', "params": {'column': 'liquidity_sum', 'window': 96, 'transform': 'log1p'}},
    {"factory": 'calendar_time_features', "params": {}},
)


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
