"""The GUI's per-bar data_source points at the Arrow store, not the old fetch."""

from __future__ import annotations

from mill import serve
from mill.bars import arrow_series
from mill.settings import PROJECT_END_STR, PROJECT_START_STR


def test_every_bar_data_source_is_get_arrow_file() -> None:
    assert set(serve._BAR_DATA_SOURCES) == set(serve._BARS)
    for bar, source in serve._BAR_DATA_SOURCES.items():
        assert source["method"] == "limen.data.HistoricalData.get_arrow_file"
        params = source["params"]
        assert isinstance(params, dict)
        assert params["file_path"] == f"/opt/arrow/{arrow_series(bar)}/latest.arrow"
        assert params["start_date_limit"] == PROJECT_START_STR
        assert params["end_date_limit"] == PROJECT_END_STR


def test_no_bar_uses_the_old_getdata_methods() -> None:
    methods = {s["method"] for s in serve._BAR_DATA_SOURCES.values()}
    assert "GetData.get_time_bars" not in methods
    assert "GetData.get_dollar_bars" not in methods


def test_data_source_path_is_canonical_not_local_override(
    monkeypatch,
) -> None:
    # The manifest path is the deploy path the Limen runner sees, independent of
    # any MILL_ARROW_DIR override used locally.
    monkeypatch.setenv("MILL_ARROW_DIR", "/tmp/somewhere/else")
    params = serve._arrow_data_source("1h_time_bars")["params"]
    assert isinstance(params, dict)
    assert params["file_path"] == "/opt/arrow/time_1h/latest.arrow"
