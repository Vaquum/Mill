from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import polars as pl

from bars import FIXED_DOLLAR_BAR_THRESHOLDS, TIME_BAR_SECONDS
from feature_registry import expr_for
from get_data import GetData

_START = "2024-01-01"
_END = "2026-05-15"


def klines(bar_name: str) -> pl.DataFrame:
    gd = GetData()
    if bar_name in TIME_BAR_SECONDS:
        return gd.get_time_bars(
            bar_name,
            start_date_limit=_START,
            end_date_limit=_END,
        )
    if bar_name in FIXED_DOLLAR_BAR_THRESHOLDS:
        df = gd.get_dollar_bars(
            bar_name,
            start_date_limit=_START,
            end_date_limit=_END,
        )
        return df.rename({"bar_start_datetime": "datetime"})
    raise ValueError(f"Unknown bar: {bar_name!r}.")


def select_features(raw: pl.DataFrame, expressions: Iterable[pl.Expr]) -> pl.DataFrame:
    return raw.select(pl.col("datetime"), *expressions)


def build_from_spec(
    raw: pl.DataFrame, spec: Sequence[dict[str, Any]]
) -> pl.DataFrame:
    exprs = [expr_for(item["factory"], item.get("params") or {}) for item in spec]
    return select_features(raw, exprs)
