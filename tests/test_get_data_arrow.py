"""GetData reads bars zero-copy from the Arrow store, every bar through the latest, normalized."""

from __future__ import annotations

import polars as pl
import pytest

from mill.bars import BAR_VALUE_COLUMNS
from mill.get_data import GetData

from conftest import ArrowStore, ns


# Bars on both sides of the window Mill once fenced itself to; all are served.
_TS = [
    ns(2023, 12, 31),
    ns(2024, 1, 2),
    ns(2025, 6, 1),
    ns(2026, 5, 15),
    ns(2026, 10, 16),
]
_DAYS = ["2023-12-31", "2024-01-02", "2025-06-01", "2026-05-15", "2026-10-16"]
_EXPECTED_COLUMNS = ["datetime", *BAR_VALUE_COLUMNS]


def test_get_time_bars_reads_every_bar(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_15m", _TS)
    out = GetData().get_time_bars("15m_time_bars")
    assert out.columns == _EXPECTED_COLUMNS
    assert out.schema["datetime"] == pl.Datetime("ms", time_zone="UTC")
    assert out["datetime"].dt.strftime("%Y-%m-%d").to_list() == _DAYS


def test_alias_resolves_to_same_series(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_15m", _TS)
    assert GetData().get_time_bars("15m").equals(
        GetData().get_time_bars("15m_time_bars")
    )


def test_get_dollar_bars_drops_index_columns(arrow_store: ArrowStore) -> None:
    # Dollar series carry start_ts and dollar_bar_id alongside ts; Mill keeps
    # only datetime + the measure columns.
    extra = {
        "start_ts": pl.Series([t - 60 for t in _TS], dtype=pl.Int64),
        "dollar_bar_id": pl.Series(list(range(len(_TS))), dtype=pl.Int64),
    }
    arrow_store.write("dollar_15M", _TS, extra=extra)
    out = GetData().get_dollar_bars("15M_dollar_bars")
    assert out.columns == _EXPECTED_COLUMNS
    assert "ts" not in out.columns
    assert "dollar_bar_id" not in out.columns
    assert out.height == len(_TS)


def test_values_are_carried_verbatim(arrow_store: ArrowStore) -> None:
    written = arrow_store.write("time_1h", _TS)
    out = GetData().get_time_bars("1h_time_bars")
    # close ramps as offset*100 + row.
    assert out["close"].to_list() == written["close"].to_list()
    assert out["close"][0] == BAR_VALUE_COLUMNS.index("close") * 100


def test_row_count_limit_returns_latest(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    out = GetData().get_time_bars("1h_time_bars", row_count_limit=2)
    days = out["datetime"].dt.strftime("%Y-%m-%d").to_list()
    assert days == ["2026-05-15", "2026-10-16"]


def test_n_rows_is_a_legacy_alias(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    assert GetData().get_time_bars("1h_time_bars", n_rows=1).height == 1


def test_row_count_limit_and_n_rows_conflict(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    with pytest.raises(ValueError):
        GetData().get_time_bars("1h_time_bars", row_count_limit=1, n_rows=1)


def test_date_limits_filter_without_bounds(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    assert GetData().get_time_bars("1h_time_bars", start_date_limit="2020-01-01").height == len(_TS)
    out = GetData().get_time_bars("1h_time_bars", start_date_limit="2025-01-01", end_date_limit="2026-06-30")
    assert out["datetime"].dt.strftime("%Y-%m-%d").to_list() == ["2025-06-01", "2026-05-15"]
    with pytest.raises(ValueError):
        GetData().get_time_bars("1h_time_bars", start_date_limit="2026-01-02", end_date_limit="2026-01-01")


def test_second_call_is_served_from_cache(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    first = GetData().get_time_bars("1h_time_bars")
    # Remove the source file; with nothing to compare, the cache serves the last known frame.
    (arrow_store.root / "time_1h").rename(arrow_store.root / "time_1h_gone")
    second = GetData().get_time_bars("1h_time_bars")
    assert first.equals(second)
