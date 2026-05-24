from __future__ import annotations

import inspect
from typing import Any, Callable

import polars as pl

import limen.features as _LF
import limen.indicators as _LI

Producer = Callable[..., pl.DataFrame]

_RAW_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "mean",
    "std",
    "maker_ratio",
    "no_of_trades",
    "open_liquidity",
    "high_liquidity",
    "low_liquidity",
    "close_liquidity",
    "liquidity_sum",
    "maker_volume",
    "maker_liquidity",
)


def _raw_producer(name: str) -> Producer:
    def run(raw: pl.DataFrame) -> pl.DataFrame:
        return raw.select(pl.col(name))
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
    return run


_TYPE_FROM_ANNOTATION: dict[type, str] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
}


def _param_type(p: inspect.Parameter) -> str | None:
    if p.default is not inspect.Parameter.empty:
        for cls, name in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
            if isinstance(p.default, cls):
                return name
        return None
    if p.annotation in _TYPE_FROM_ANNOTATION:
        return _TYPE_FROM_ANNOTATION[p.annotation]
    return None


def _limen_param_specs(fn: Callable[..., Any]) -> tuple[dict[str, Any], ...]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ()
    out: list[dict[str, Any]] = []
    for p in sig.parameters.values():
        if p.name in {"data", "df"}:
            continue
        if p.name == "output_col" or p.name.endswith("_col"):
            continue
        type_ = _param_type(p)
        if type_ is None:
            continue
        default = p.default if p.default is not inspect.Parameter.empty else None
        out.append({"name": p.name, "type": type_, "default": default})
    return tuple(out)


REGISTRY: dict[str, dict[str, Any]] = {}

for _name in _RAW_COLUMNS:
    REGISTRY[_name] = {
        "producer": _raw_producer(_name),
        "params": (),
        "source": "raw",
    }


def _register_limen(ns: Any, source: str) -> None:
    for name in dir(ns):
        if name.startswith("_"):
            continue
        if name in REGISTRY:
            continue
        fn = getattr(ns, name)
        if not callable(fn):
            continue
        REGISTRY[name] = {
            "producer": _limen_producer(fn),
            "params": _limen_param_specs(fn),
            "source": source,
        }


_register_limen(_LF, "limen.features")
_register_limen(_LI, "limen.indicators")


def producer_for(factory_name: str) -> Producer:
    if factory_name not in REGISTRY:
        raise ValueError(f"Unknown factory: {factory_name!r}.")
    return REGISTRY[factory_name]["producer"]


def catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "params": list(entry["params"]),
            "source": entry["source"],
        }
        for name, entry in REGISTRY.items()
    ]
