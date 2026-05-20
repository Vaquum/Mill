from __future__ import annotations

import polars as pl

from table import build_from_spec

NAME = 'gen3'

TARGET = 'abs_vol_normalized_return'

SPEC = (
    {"factory": 'parkinson_volatility', "params": {'n': 1}},
    {"factory": 'parkinson_volatility', "params": {'n': 4}},
    {"factory": 'liquidity_z', "params": {'n': 96}},
    {"factory": 'is_weekend', "params": {}},
    {"factory": 'nth_hour', "params": {}},
)


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
