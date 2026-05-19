from __future__ import annotations

import polars as pl

from table import build_from_spec

NAME = "roc"

SPEC = tuple(
    {"factory": "roc", "params": {"n": n}}
    for n in (1, 2, 4, 8, 16, 32, 64, 128)
)


def build(raw: pl.DataFrame) -> pl.DataFrame:
    return build_from_spec(raw, SPEC)
