from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import polars as pl

from bars import DOLLAR_BAR_USD, TIME_BAR_SECONDS
from feature_registry import producer_for
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
    if bar_name in DOLLAR_BAR_USD:
        return gd.get_dollar_bars(
            bar_name,
            start_date_limit=_START,
            end_date_limit=_END,
        )
    raise ValueError(f"Unknown bar: {bar_name!r}.")


def build_from_spec(
    raw: pl.DataFrame, spec: Sequence[dict[str, Any]]
) -> pl.DataFrame:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for item in spec:
        params = item.get("params") or {}
        key = (item["factory"], json.dumps(params, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    parts: list[pl.DataFrame] = [raw.select(pl.col("datetime"))]
    for item in unique:
        producer = producer_for(item["factory"])
        parts.append(producer(raw, **(item.get("params") or {})))
    return pl.concat(parts, how="horizontal")
