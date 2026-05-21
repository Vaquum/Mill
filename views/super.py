from __future__ import annotations

import polars as pl

from table import build_from_spec

NAME = 'super'

TARGET = 'abs_vol_normalized_return'

SPEC = (
    {"factory": 'rolling_zscore', "params": {'column': 'close', 'window': 96, 'transform': 'abs'}},
    {"factory": 'atr_sma', "params": {'period': 14}},
    {"factory": 'calendar_time_features', "params": {}},
    {"factory": 'rolling_zscore', "params": {'column': 'liquidity_sum', 'window': 96, 'transform': 'log1p'}},
    {"factory": 'narrow_range', "params": {'window': 24}},
    {"factory": 'rolling_zscore', "params": {'column': 'no_of_trades', 'window': 96, 'transform': 'log1p'}},
    {"factory": 'volatility_term_structure', "params": {'short_window': 12, 'medium_window': 48, 'long_window': 168}},
    {"factory": 'volume_regime', "params": {'lookback': 48}},
    {"factory": 'rolling_zscore', "params": {'column': 'volume', 'window': 96, 'transform': 'log1p'}},
)


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
