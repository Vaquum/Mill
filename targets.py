from __future__ import annotations

import inspect
import re
from collections.abc import Callable

import polars as pl

import limen.targets as _LT


def _snake_case(class_name: str) -> str:
    stem = class_name[:-len("Target")] if class_name.endswith("Target") else class_name
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()


def _is_instantiable(cls: type) -> bool:
    sig = inspect.signature(cls.__init__)
    for name, p in sig.parameters.items():
        if name in {"self", "train_data", "target_name"}:
            continue
        if p.default is inspect.Parameter.empty and p.kind not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            return False
    return True


def _bind(cls: type, **kwargs: object) -> Callable[[pl.DataFrame, str], pl.DataFrame]:
    def fn(raw: pl.DataFrame, name: str) -> pl.DataFrame:
        return cls(train_data=raw, target_name=name, **kwargs).transform(raw)
    return fn


_PROBE = pl.DataFrame({
    "datetime": pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 8), interval="1h", eager=True
    ),
    "open": [100.0] * 169,
    "high": [101.0] * 169,
    "low": [99.0] * 169,
    "close": [100.5] * 169,
    "volume": [10.0] * 169,
    "maker_ratio": [0.5] * 169,
    "no_of_trades": [100] * 169,
    "liquidity_sum": [1.0e6] * 169,
    "maker_volume": [5.0] * 169,
    "maker_liquidity": [5.0e5] * 169,
    "high_liquidity": [1.1e6] * 169,
    "low_liquidity": [9.0e5] * 169,
})


def _can_apply(cls: type) -> bool:
    try:
        cls(train_data=_PROBE, target_name="_probe").transform(_PROBE)
    except Exception:
        return False
    return True


_FACTORIES: dict[str, Callable[[pl.DataFrame, str], pl.DataFrame]] = {
    _snake_case(name): _bind(cls)
    for name in _LT.__all__
    if _is_instantiable(cls := getattr(_LT, name)) and _can_apply(cls)
}
_FACTORIES["abs_forward_vol_normalized_return"] = _bind(
    _LT.ForwardVolNormalizedReturnTarget, absolute=True
)

NAMES: tuple[str, ...] = tuple(_FACTORIES)


def apply(raw: pl.DataFrame, name: str) -> pl.DataFrame:
    if name not in _FACTORIES:
        raise ValueError(f"Unknown target: {name!r}.")
    return _FACTORIES[name](raw, name)
