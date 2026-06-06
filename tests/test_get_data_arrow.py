"""GetData reads bars zero-copy from the Arrow store, windowed and normalized."""

from __future__ import annotations

import polars as pl
import pytest

from mill.bars import BAR_VALUE_COLUMNS
from mill.get_data import GetData

from conftest import ArrowStore, ns


# 2023-12-31 is before the project window, 2026-06-01 after it; the three
# middle days are in-window (2026-05-15 is the inclusive end).
_TS = [
    ns(2023, 12, 31),
    ns(2024, 1, 2),
    ns(2025, 6, 1),
    ns(2026, 5, 15),
    ns(2026, 6, 1),
]
_IN_WINDOW = 3
_EXPECTED_COLUMNS = ["datetime", *BAR_VALUE_COLUMNS]


def test_get_time_bars_reads_and_windows(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_15m", _TS)
    out = GetData().get_time_bars("15m_time_bars")
    assert out.columns == _EXPECTED_COLUMNS
    assert out.height == _IN_WINDOW
    assert out.schema["datetime"] == pl.Datetime("ms", time_zone="UTC")
    days = out["datetime"].dt.strftime("%Y-%m-%d").to_list()
    assert days == ["2024-01-02", "2025-06-01", "2026-05-15"]


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
    assert out.height == _IN_WINDOW


def test_values_are_carried_verbatim(arrow_store: ArrowStore) -> None:
    written = arrow_store.write("time_1h", _TS)
    out = GetData().get_time_bars("1h_time_bars")
    # close ramps as offset*100 + row; row 1 (2024-01-02) is the first in-window.
    close_offset = BAR_VALUE_COLUMNS.index("close") * 100
    assert out["close"].to_list() == [
        written["close"][1],
        written["close"][2],
        written["close"][3],
    ]
    assert out["close"][0] == close_offset + 1


def test_row_count_limit_returns_latest(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    out = GetData().get_time_bars("1h_time_bars", row_count_limit=2)
    days = out["datetime"].dt.strftime("%Y-%m-%d").to_list()
    assert days == ["2025-06-01", "2026-05-15"]


def test_n_rows_is_a_legacy_alias(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    assert GetData().get_time_bars("1h_time_bars", n_rows=1).height == 1


def test_row_count_limit_and_n_rows_conflict(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    with pytest.raises(ValueError):
        GetData().get_time_bars("1h_time_bars", row_count_limit=1, n_rows=1)


def test_out_of_bounds_dates_rejected(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    with pytest.raises(ValueError):
        GetData().get_time_bars("1h_time_bars", start_date_limit="2020-01-01")


def test_second_call_is_served_from_cache(arrow_store: ArrowStore) -> None:
    arrow_store.write("time_1h", _TS)
    first = GetData().get_time_bars("1h_time_bars")
    # Remove the source file; a cache hit must still serve the windowed frame.
    (arrow_store.root / "time_1h").rename(arrow_store.root / "time_1h_gone")
    second = GetData().get_time_bars("1h_time_bars")
    assert first.equals(second)
