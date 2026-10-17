from __future__ import annotations

import inspect

import polars as pl

import limen.targets as _LT
from .memo import ByteLRU, Flights
from .settings import memory_budget

_RESULTS = ByteLRU(memory_budget("targets", 16), lambda frame: frame.estimated_size())
_FLIGHTS = Flights()


_BINDINGS: dict[str, tuple[type, dict[str, object]]] = {
    "ema_breakout": (_LT.EmaBreakoutTarget, {}),
    "forward_breakout": (_LT.ForwardBreakoutTarget, {}),
    "forward_vol_normalized_return": (_LT.ForwardVolNormalizedReturnTarget, {}),
    "next_bar_down": (_LT.NextBarDownTarget, {}),
    "next_bar_up": (_LT.NextBarUpTarget, {}),
    "next_return": (_LT.NextReturnTarget, {}),
    "abs_forward_vol_normalized_return": (
        _LT.ForwardVolNormalizedReturnTarget,
        {"absolute": True},
    ),
}

NAMES: tuple[str, ...] = tuple(_BINDINGS)


def _default_params(cls: type) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, p in inspect.signature(cls.__init__).parameters.items():
        if name in {"self", "train_data", "target_name"}:
            continue
        if p.default is inspect.Parameter.empty:
            continue
        out[name] = p.default
    return out


def apply(raw: pl.DataFrame, name: str) -> pl.DataFrame:
    if name not in _BINDINGS:
        raise ValueError(f"Unknown target: {name!r}.")
    cls, kwargs = _BINDINGS[name]
    return cls(train_data=raw, target_name=name, **kwargs).transform(raw)


def for_source(raw, name, source_key):
    """A target depends on the source snapshot, not on a view's feature spec."""
    if source_key is None:
        return apply(raw, name).select("datetime", name)
    key = (source_key, name)
    with _FLIGHTS.hold(key):
        result = _RESULTS.get(key)
        if result is None:
            result = apply(raw, name).select("datetime", name)
            _RESULTS[key] = result
        return result.clone()


def binding(name: str) -> dict[str, object]:
    if name not in _BINDINGS:
        raise ValueError(f"Unknown target: {name!r}.")
    cls, kwargs = _BINDINGS[name]
    params = _default_params(cls)
    params.update(kwargs)
    return {"class": f"limen.targets.{cls.__name__}", "params": params}
