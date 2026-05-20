from __future__ import annotations

import polars as pl

from table import build_from_spec

NAME = "jolt"

SPEC = (
    {"factory": "realized_volatility",            "params": {"n": 4}},
    {"factory": "realized_volatility",            "params": {"n": 16}},
    {"factory": "realized_semivolatility_down",   "params": {"n": 4}},
    {"factory": "realized_semivolatility_down",   "params": {"n": 16}},
    {"factory": "parkinson_volatility",           "params": {"window": 1}},
    {"factory": "parkinson_volatility",           "params": {"window": 4}},
    {"factory": "jump_volatility",                "params": {"n": 16}},
    {"factory": "liquidity_z",                    "params": {"n": 96}},
    {"factory": "is_funding_hour",                "params": {}},
    {"factory": "is_us_open_hour",                "params": {}},
    {"factory": "is_weekend",                     "params": {}},
)


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
