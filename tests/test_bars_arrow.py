"""Every Mill bar resolves to exactly one tdw Arrow bar-store series."""

from __future__ import annotations

import pytest

from mill import bars
from mill.feature_registry import _RAW_COLUMNS


def test_every_bar_maps_to_a_series() -> None:
    for bar in bars.BAR_NAMES:
        assert bar in bars.BAR_ARROW_SERIES
    assert set(bars.BAR_ARROW_SERIES) == set(bars.BAR_NAMES)


def test_series_names_match_the_tdw_store() -> None:
    # The series directory names are the contract with the tdw-control-plane
    # Arrow store (build_bar_store_arrow SPECS); pin them literally.
    assert bars.TIME_BAR_ARROW_SERIES == {
        "15m_time_bars": "time_15m",
        "30m_time_bars": "time_30m",
        "1h_time_bars": "time_1h",
        "2h_time_bars": "time_2h",
        "4h_time_bars": "time_4h",
    }
    assert bars.DOLLAR_BAR_ARROW_SERIES == {
        "15M_dollar_bars": "dollar_15M",
        "30M_dollar_bars": "dollar_30M",
        "60M_dollar_bars": "dollar_60M",
        "120M_dollar_bars": "dollar_120M",
        "240M_dollar_bars": "dollar_240M",
    }


def test_series_helpers_accept_aliases() -> None:
    assert bars.time_bar_series("15m") == "time_15m"
    assert bars.time_bar_series("4h_time_bars") == "time_4h"
    assert bars.time_bar_series(3600) == "time_1h"
    assert bars.dollar_bar_series("60M") == "dollar_60M"
    assert bars.dollar_bar_series("240M_dollar_bars") == "dollar_240M"


def test_arrow_series_rejects_unknown_bar() -> None:
    with pytest.raises(ValueError):
        bars.arrow_series("nope")


def test_value_columns_are_the_feature_registry_raw_columns() -> None:
    # feature_registry sources its raw factories from bars.BAR_VALUE_COLUMNS, so
    # the columns Mill keeps from the store are exactly the raw feature sources.
    assert tuple(_RAW_COLUMNS) == bars.BAR_VALUE_COLUMNS
    assert len(bars.BAR_VALUE_COLUMNS) == 16
