from __future__ import annotations

import math
from collections.abc import Callable

import polars as pl

from limen.targets import NextBarDownTarget, NextBarUpTarget


_PARKINSON_SCALE = 4.0 * math.log(2.0)
_HALFLIFE = 50
_MIN_PERIODS = 150
_SIGMA_RATIO_MIN = 0.9
_SIGMA_RATIO_MAX = 1.1


def _from_limen(cls: type) -> Callable[[pl.DataFrame, str], pl.DataFrame]:
    def fn(raw: pl.DataFrame, name: str) -> pl.DataFrame:
        return cls(train_data=raw, target_name=name).transform(raw)
    return fn


def _validate_ohlc(df: pl.DataFrame) -> None:
    dirty = df.filter(
        (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
    ).height
    if dirty:
        raise ValueError(
            f"vol_normalized_return: {dirty} bar(s) violate OHLC invariants; "
            "refusing to silently filter."
        )


def _validate_sanity(df: pl.DataFrame) -> None:
    close_var = (pl.col("close").log() - pl.col("close").shift(1).log()) ** 2
    parkinson_var = (
        (pl.col("high").log() - pl.col("low").log()) ** 2 / _PARKINSON_SCALE
    )
    ratio = (
        df.with_columns([close_var.alias("_cv"), parkinson_var.alias("_pv")])
        .with_columns([
            pl.col("_cv")
                .ewm_mean(half_life=_HALFLIFE, min_samples=_MIN_PERIODS)
                .clip(lower_bound=0.0).sqrt().alias("_cs"),
            pl.col("_pv")
                .ewm_mean(half_life=_HALFLIFE, min_samples=_MIN_PERIODS)
                .clip(lower_bound=0.0).sqrt().alias("_ps"),
        ])
        .with_columns(
            pl.when(pl.col("_cs") == 0.0)
                .then(None)
                .otherwise(pl.col("_ps") / pl.col("_cs"))
                .alias("_ratio")
        )
        .select(pl.col("_ratio").drop_nulls().median())
        .item()
    )
    if ratio is None:
        return
    if not (_SIGMA_RATIO_MIN <= ratio <= _SIGMA_RATIO_MAX):
        raise ValueError(
            f"vol_normalized_return: median Parkinson/close-σ ratio {ratio:.6f} "
            f"outside [{_SIGMA_RATIO_MIN}, {_SIGMA_RATIO_MAX}]."
        )


def _vol_scaled_forward(
    raw: pl.DataFrame, name: str, *, absolute: bool
) -> pl.DataFrame:
    _validate_ohlc(raw)
    _validate_sanity(raw)

    parkinson_var = (
        (pl.col("high").log() - pl.col("low").log()) ** 2 / _PARKINSON_SCALE
    )
    df = raw.with_columns(parkinson_var.alias("_p_var"))
    df = df.with_columns(
        pl.col("_p_var")
            .ewm_mean(half_life=_HALFLIFE, min_samples=_MIN_PERIODS)
            .clip(lower_bound=0.0)
            .sqrt()
            .alias("_sigma")
    )
    fwd = pl.col("close").shift(-1).log() - pl.col("close").log()
    df = df.with_columns((fwd.abs() if absolute else fwd).alias("_fwd"))
    df = df.with_columns(
        pl.when(pl.col("_sigma") == 0.0)
            .then(None)
            .otherwise(pl.col("_fwd") / pl.col("_sigma"))
            .alias(name)
    )
    return df.drop("_p_var", "_sigma", "_fwd")


def _vol_normalized_return(raw: pl.DataFrame, name: str) -> pl.DataFrame:
    return _vol_scaled_forward(raw, name, absolute=False)


def _abs_vol_normalized_return(raw: pl.DataFrame, name: str) -> pl.DataFrame:
    return _vol_scaled_forward(raw, name, absolute=True)


_FACTORIES: dict[str, Callable[[pl.DataFrame, str], pl.DataFrame]] = {
    "next_bar_up": _from_limen(NextBarUpTarget),
    "next_bar_down": _from_limen(NextBarDownTarget),
    "vol_normalized_return": _vol_normalized_return,
    "abs_vol_normalized_return": _abs_vol_normalized_return,
}

NAMES: tuple[str, ...] = tuple(_FACTORIES)


def apply(raw: pl.DataFrame, name: str) -> pl.DataFrame:
    if name not in _FACTORIES:
        raise ValueError(f"Unknown target: {name!r}.")
    return _FACTORIES[name](raw, name)
