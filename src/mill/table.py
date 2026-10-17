from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any

import polars as pl

from .bars import DOLLAR_BAR_USD, TIME_BAR_SECONDS
from .feature_registry import producer_for
from .get_data import GetData
from .expressions import evaluate
from .memo import ByteLRU, Flights
from .settings import memory_budget

_PRODUCERS = ByteLRU(memory_budget("producers", 256), lambda frame: frame.estimated_size())
_PRODUCER_FLIGHTS = Flights()


def produce(raw: pl.DataFrame, factory: str, params: dict[str, Any], cache_key=None) -> pl.DataFrame:
    if cache_key is None:
        return producer_for(factory)(raw, **params)
    key = (cache_key, factory, json.dumps(params, sort_keys=True))
    with _PRODUCER_FLIGHTS.hold(key):
        hit = _PRODUCERS.get(key)
        if hit is None:
            hit = producer_for(factory)(raw, **params)
            _PRODUCERS[key] = hit
        return hit.clone()


def produce_many(raw, spec, cache_key=None):
    """One dependency graph for uncached producers, with existing per-key exclusion."""
    requests = [(producer_for(item["factory"]), item.get("params") or {}) for item in spec]
    if cache_key is None:
        return evaluate(raw, requests)
    keys = [(cache_key, item["factory"], json.dumps(params, sort_keys=True))
            for item, (_, params) in zip(spec, requests)]
    with ExitStack() as stack:
        for key in sorted(set(keys), key=repr):
            stack.enter_context(_PRODUCER_FLIGHTS.hold(key))
        results = [_PRODUCERS.get(key) for key in keys]
        missing = [i for i, result in enumerate(results) if result is None]
        if missing:
            computed = evaluate(raw, [requests[i] for i in missing])
            for i, result in zip(missing, computed):
                _PRODUCERS[keys[i]] = results[i] = result
        return [result.clone() for result in results]


def klines(bar_name: str) -> pl.DataFrame:
    """A bar series as the store holds it, through its latest bar."""
    gd = GetData()
    if bar_name in TIME_BAR_SECONDS:
        return gd.get_time_bars(bar_name)
    if bar_name in DOLLAR_BAR_USD:
        return gd.get_dollar_bars(bar_name)
    raise ValueError(f"Unknown bar: {bar_name!r}.")


def build_from_spec(
    raw: pl.DataFrame, spec: Sequence[dict[str, Any]], *, cache_key=None
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
    parts.extend(produce_many(raw, unique, cache_key))
    return pl.concat(parts, how="horizontal")
