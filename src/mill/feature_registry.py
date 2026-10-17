from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Callable

import polars as pl

import limen.features as _LF
import limen.indicators as _LI

from .bars import BAR_VALUE_COLUMNS as _RAW_COLUMNS

Producer = Callable[..., pl.DataFrame]


def _raw_producer(name: str) -> Producer:
    def run(raw: pl.DataFrame) -> pl.DataFrame:
        return raw.select(pl.col(name))
    run._mill_native_plan = True
    return run


def _limen_producer(fn: Callable[..., pl.DataFrame]) -> Producer:
    def run(raw: pl.DataFrame, **params: Any) -> pl.DataFrame:
        out = fn(raw, **params)
        new = [c for c in out.columns if c not in raw.columns]
        if not new:
            return out.select([])
        if not params:
            return out.select(new)
        suffix = "__" + "_".join(f"{k}={params[k]}" for k in sorted(params))
        return out.select(new).rename({c: f"{c}{suffix}" for c in new})
    run._mill_native_plan = True
    return run


def _registry() -> dict[str, dict[str, Any]]:
    """Keep Mill's existing API independent of upstream export/type-hint changes."""
    contract = json.loads(files("mill").joinpath("factory_contract.json").read_text())
    namespaces = {"limen.features": _LF, "limen.indicators": _LI}
    registry = {}
    for item in contract["factories"]:
        name, source = item["name"], item["source"]
        if source == "raw":
            if name not in _RAW_COLUMNS:
                raise ValueError(f"unknown contracted raw column: {name}")
            producer = _raw_producer(name)
        else:
            fn = getattr(namespaces[source], name)
            if not callable(fn):
                raise TypeError(f"contracted factory is not callable: {name}")
            producer = _limen_producer(fn)
        registry[name] = {"producer": producer,
            "params": tuple(dict(p) for p in item["params"]), "source": source}
    return registry


REGISTRY: dict[str, dict[str, Any]] = _registry()


def producer_for(factory_name: str) -> Producer:
    if factory_name not in REGISTRY:
        raise ValueError(f"Unknown factory: {factory_name!r}.")
    return REGISTRY[factory_name]["producer"]


def params_for(factory_name: str) -> tuple[dict[str, Any], ...]:
    if factory_name not in REGISTRY:
        raise ValueError(f"Unknown factory: {factory_name!r}.")
    return tuple(dict(p) for p in REGISTRY[factory_name]["params"])


def catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "params": [dict(p) for p in entry["params"]],
            "source": entry["source"],
        }
        for name, entry in REGISTRY.items()
    ]
