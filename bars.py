from __future__ import annotations


TIME_BAR_SECONDS = {
    "15m_time_bars": 900,
    "30m_time_bars": 1800,
    "1h_time_bars": 3600,
    "2h_time_bars": 7200,
    "4h_time_bars": 14400,
}
DOLLAR_BAR_USD = {
    "15M_dollar_bars": 15_000_000,
    "30M_dollar_bars": 30_000_000,
    "60M_dollar_bars": 60_000_000,
    "120M_dollar_bars": 120_000_000,
    "240M_dollar_bars": 240_000_000,
}
BAR_NAMES = tuple(TIME_BAR_SECONDS) + tuple(DOLLAR_BAR_USD)

_TIME_ALIASES = {
    "15m": "15m_time_bars",
    "30m": "30m_time_bars",
    "1h": "1h_time_bars",
    "2h": "2h_time_bars",
    "4h": "4h_time_bars",
}
_DOLLAR_ALIASES = {
    "15M": "15M_dollar_bars",
    "30M": "30M_dollar_bars",
    "60M": "60M_dollar_bars",
    "120M": "120M_dollar_bars",
    "240M": "240M_dollar_bars",
}
_SECONDS_TO_NAME = {seconds: name for name, seconds in TIME_BAR_SECONDS.items()}


def time_bar_name(bar_size: str | int) -> str:
    if isinstance(bar_size, int):
        if bar_size in _SECONDS_TO_NAME:
            return _SECONDS_TO_NAME[bar_size]
        raise ValueError(f"Unsupported time bar size: {bar_size}.")
    if bar_size in TIME_BAR_SECONDS:
        return bar_size
    if bar_size in _TIME_ALIASES:
        return _TIME_ALIASES[bar_size]
    raise ValueError(f"Unsupported time bar size: {bar_size}.")


def time_bar_seconds(bar_size: str | int) -> int:
    return TIME_BAR_SECONDS[time_bar_name(bar_size)]


def dollar_bar_name(bar_size: str) -> str:
    if bar_size in DOLLAR_BAR_USD:
        return bar_size
    if bar_size in _DOLLAR_ALIASES:
        return _DOLLAR_ALIASES[bar_size]
    raise ValueError(f"Unsupported dollar bar size: {bar_size}.")


def dollar_bar_usd(bar_size: str) -> int:
    return DOLLAR_BAR_USD[dollar_bar_name(bar_size)]
