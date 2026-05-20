from __future__ import annotations

import polars as pl

from table import build_from_spec

NAME = 'gen4'

TARGET = 'abs_vol_normalized_return'

SPEC = (
    {"factory": 'parkinson_volatility', "params": {'window': 1}},
    {"factory": 'parkinson_volatility', "params": {'window': 4}},
    {"factory": 'liquidity_z', "params": {'n': 96}},
    {"factory": 'is_weekend', "params": {}},
    {"factory": 'nth_hour', "params": {}},
    # New edge features
    {"factory": 'abs_price_z', "params": {'n': 96}},
    {"factory": 'taker_imbalance_abs_z', "params": {'n': 96}},
    {"factory": 'high_low_range_trend', "params": {'n': 24}},
    {"factory": 'volatility_ratio', "params": {'short_n': 4, 'long_n': 96}},
)

def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
