from __future__ import annotations

import polars as pl


TIME_BAR_SECONDS = {
    "15m_time_bars": 900,
    "30m_time_bars": 1800,
    "1h_time_bars": 3600,
    "2h_time_bars": 7200,
    "4h_time_bars": 14400,
}
DOLLAR_BAR_ANCHORS = {
    "15m_dollar_bars": "15m_time_bars",
    "30m_dollar_bars": "30m_time_bars",
    "1h_dollar_bars": "1h_time_bars",
    "2h_dollar_bars": "2h_time_bars",
    "4h_dollar_bars": "4h_time_bars",
}
FIXED_DOLLAR_BAR_THRESHOLDS = {
    "20M_dollar_bars": 20_000_000.0,
    "40M_dollar_bars": 40_000_000.0,
    "80M_dollar_bars": 80_000_000.0,
    "160M_dollar_bars": 160_000_000.0,
    "320M_dollar_bars": 320_000_000.0,
}
BAR_NAMES = (
    tuple(TIME_BAR_SECONDS)
    + tuple(DOLLAR_BAR_ANCHORS)
    + tuple(FIXED_DOLLAR_BAR_THRESHOLDS)
)
_SECONDS_TO_NAME = {seconds: name for name, seconds in TIME_BAR_SECONDS.items()}
_ALIASES = {
    "15m": "15m_time_bars",
    "30m": "30m_time_bars",
    "1h": "1h_time_bars",
    "2h": "2h_time_bars",
    "4h": "4h_time_bars",
}
_DOLLAR_ALIASES = {
    "15m_dollar": "15m_dollar_bars",
    "30m_dollar": "30m_dollar_bars",
    "1h_dollar": "1h_dollar_bars",
    "2h_dollar": "2h_dollar_bars",
    "4h_dollar": "4h_dollar_bars",
}

BAR_COLUMNS = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "mean",
    "std",
    "volume",
    "maker_ratio",
    "no_of_trades",
    "open_liquidity",
    "high_liquidity",
    "low_liquidity",
    "close_liquidity",
    "liquidity_sum",
    "maker_volume",
    "maker_liquidity",
]


def time_bar_name(bar_size: str | int) -> str:
    if isinstance(bar_size, int):
        if bar_size in _SECONDS_TO_NAME:
            return _SECONDS_TO_NAME[bar_size]
        raise ValueError(f"Unsupported time bar size: {bar_size}.")

    if bar_size in TIME_BAR_SECONDS:
        return bar_size
    if bar_size in _ALIASES:
        return _ALIASES[bar_size]

    raise ValueError(f"Unsupported time bar size: {bar_size}.")


def time_bar_seconds(bar_size: str | int) -> int:
    return TIME_BAR_SECONDS[time_bar_name(bar_size)]


def dollar_bar_name(bar_size: str) -> str:
    if bar_size in DOLLAR_BAR_ANCHORS:
        return bar_size
    if bar_size in FIXED_DOLLAR_BAR_THRESHOLDS:
        return bar_size
    if bar_size in _DOLLAR_ALIASES:
        return _DOLLAR_ALIASES[bar_size]
    raise ValueError(f"Unsupported dollar bar size: {bar_size}.")


def round_time_bars(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns([
        pl.col("mean").round(5),
        pl.col("std").round(6),
        pl.col("volume").round(9),
        pl.col("liquidity_sum").round(1),
        pl.col("maker_liquidity").round(1),
    ])


def time_bars_from_1m(data: pl.DataFrame, bar_seconds: int) -> pl.DataFrame:
    missing = sorted(set(BAR_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError("Time bar aggregation missing columns: " + ", ".join(missing))

    data = data.sort("datetime")
    if bar_seconds < 60 or bar_seconds % 60 != 0:
        raise ValueError("time bar size must be 60 seconds or a multiple of 60.")
    if bar_seconds == 60:
        return round_time_bars(data.select(BAR_COLUMNS))

    bucket_ms = bar_seconds * 1000
    weighted = data.with_columns([
        (
            pl.col("datetime")
            .dt.epoch("ms")
            .floordiv(bucket_ms)
            .mul(bucket_ms)
            .cast(pl.Datetime("ms", time_zone="UTC"))
        ).alias("_bucket_datetime"),
        (pl.col("mean") * pl.col("no_of_trades").cast(pl.Float64)).alias(
            "_mean_weighted_sum"
        ),
        (
            (pl.col("std").pow(2) + pl.col("mean").pow(2))
            * pl.col("no_of_trades").cast(pl.Float64)
        ).alias("_sum_of_squares"),
        (pl.col("maker_ratio") * pl.col("no_of_trades").cast(pl.Float64)).alias(
            "_maker_ratio_weighted_sum"
        ),
    ])

    no_of_trades_sum = pl.col("no_of_trades").sum().cast(pl.Float64)
    mean_expr = pl.col("_mean_weighted_sum").sum() / no_of_trades_sum
    variance_expr = (pl.col("_sum_of_squares").sum() / no_of_trades_sum) - mean_expr.pow(2)

    aggregated = (
        weighted.group_by("_bucket_datetime", maintain_order=True)
        .agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            mean_expr.alias("mean"),
            pl.when(variance_expr < 0).then(0.0).otherwise(variance_expr).sqrt().alias("std"),
            pl.col("volume").sum().alias("volume"),
            (pl.col("_maker_ratio_weighted_sum").sum() / no_of_trades_sum).alias(
                "maker_ratio"
            ),
            pl.col("no_of_trades").sum().alias("no_of_trades"),
            pl.col("open_liquidity").first().alias("open_liquidity"),
            pl.col("high_liquidity").max().alias("high_liquidity"),
            pl.col("low_liquidity").min().alias("low_liquidity"),
            pl.col("close_liquidity").last().alias("close_liquidity"),
            pl.col("liquidity_sum").sum().alias("liquidity_sum"),
            pl.col("maker_volume").sum().alias("maker_volume"),
            pl.col("maker_liquidity").sum().alias("maker_liquidity"),
        ])
        .rename({"_bucket_datetime": "datetime"})
        .select(BAR_COLUMNS)
        .sort("datetime")
    )

    return round_time_bars(aggregated)
