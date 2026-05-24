from __future__ import annotations

import polars as pl

from ..table import build_from_spec

NAME = "ochlv"

SPEC = (
    {"factory": "open",   "params": {}},
    {"factory": "close",  "params": {}},
    {"factory": "high",   "params": {}},
    {"factory": "low",    "params": {}},
    {"factory": "volume", "params": {}},
)


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
