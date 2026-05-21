from __future__ import annotations

import polars as pl

from table import build_from_spec

NAME = 'gen4'

TARGET = 'abs_vol_normalized_return'

SPEC = (
    {"factory": 'parkinson_volatility', "params": {'window': 1}},
    {"factory": 'parkinson_volatility', "params": {'window': 4}},
    {"factory": 'rolling_zscore', "params": {'column': 'liquidity_sum', 'window': 96, 'transform': 'log1p'}},
    {"factory": 'calendar_time_features', "params": {}},
    {"factory": 'rolling_zscore', "params": {'column': 'close', 'window': 96, 'transform': 'abs'}},
    {"factory": 'taker_imbalance_ratio', "params": {'window': 96}},
    {"factory": 'narrow_range', "params": {'window': 24}},
    {"factory": 'volatility_ratio', "params": {'short_window': 4, 'long_window': 96}},
)

def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
