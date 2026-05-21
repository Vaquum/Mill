from __future__ import annotations

from collections.abc import Callable

import polars as pl

from limen.targets import (
    ForwardVolNormalizedReturnTarget,
    NextBarDownTarget,
    NextBarUpTarget,
)


def _from_limen(cls: type, **kwargs: object) -> Callable[[pl.DataFrame, str], pl.DataFrame]:
    def fn(raw: pl.DataFrame, name: str) -> pl.DataFrame:
        return cls(train_data=raw, target_name=name, **kwargs).transform(raw)
    return fn


_FACTORIES: dict[str, Callable[[pl.DataFrame, str], pl.DataFrame]] = {
    "next_bar_up": _from_limen(NextBarUpTarget),
    "next_bar_down": _from_limen(NextBarDownTarget),
    "vol_normalized_return": _from_limen(ForwardVolNormalizedReturnTarget, absolute=False),
    "abs_vol_normalized_return": _from_limen(ForwardVolNormalizedReturnTarget, absolute=True),
}

NAMES: tuple[str, ...] = tuple(_FACTORIES)


def apply(raw: pl.DataFrame, name: str) -> pl.DataFrame:
    if name not in _FACTORIES:
        raise ValueError(f"Unknown target: {name!r}.")
    return _FACTORIES[name](raw, name)
