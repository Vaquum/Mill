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
    "2M_dollar_bars": 2_000_000.0,
    "4M_dollar_bars": 4_000_000.0,
    "8M_dollar_bars": 8_000_000.0,
    "16M_dollar_bars": 16_000_000.0,
    "32M_dollar_bars": 32_000_000.0,
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
