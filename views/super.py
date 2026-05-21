from __future__ import annotations

import polars as pl

from table import build_from_spec

NAME = 'super'

TARGET = 'abs_vol_normalized_return'

SPEC = (
    {"factory": 'abs_price_z', "params": {'n': 96}},
    {"factory": 'atr_sma', "params": {'period': 14}},
    {"factory": 'is_weekend', "params": {}},
    {"factory": 'liquidity_z', "params": {'n': 96}},
    {"factory": 'narrow_range', "params": {'n': 24}},
    {"factory": 'trades_z', "params": {'n': 96}},
    {"factory": 'volatility_term_structure', "params": {'short_window': 12, 'medium_window': 48, 'long_window': 168}},
    {"factory": 'volume_regime', "params": {'lookback': 48}},
    {"factory": 'volume_z', "params": {'n': 96}},
)


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
