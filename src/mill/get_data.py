from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl
from limen import HistoricalData

from .bars import (
    BAR_VALUE_COLUMNS,
    dollar_bar_name,
    dollar_bar_series,
    time_bar_name,
    time_bar_series,
)
from .cache_store import cache_path, read_fresh, write_cache
from .frames import normalize_datetime, read_parquet, write_parquet
from .settings import arrow_latest_path


_LIMEN_LOCK = threading.Lock()


def _date_part(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError as exc:
        raise ValueError(f"Invalid date limit: {value!r}.") from exc


def _date_limits(
    start_date_limit: str | None,
    end_date_limit: str | None,
) -> tuple[str | None, str | None]:
    """Optional inclusive day limits; without them a series is read through its latest bar."""
    start, end = _date_part(start_date_limit), _date_part(end_date_limit)
    if start is not None and end is not None and start > end:
        raise ValueError("start_date_limit cannot be after end_date_limit.")
    return (
        start.isoformat() if start is not None else None,
        end.isoformat() if end is not None else None,
    )


def _limit_rows(data: pl.DataFrame, limit: int | None) -> pl.DataFrame:
    return data.tail(limit) if limit is not None else data


class GetData:
    def __init__(self) -> None:
        self._hd = HistoricalData()
        self.data = pl.DataFrame()
        self.data_columns: list[str] = []

    def from_parquet(self, path: str | Path) -> pl.DataFrame:
        return self._store(read_parquet(path))

    def to_parquet(self, path: str | Path) -> Path:
        return write_parquet(self.data, path)

    def _load_arrow(
        self,
        series: str,
        start_date_limit: str | None,
        end_date_limit: str | None,
    ) -> pl.DataFrame:
        """Zero-copy read one bar series from the tdw Arrow bar store.

        Memory-maps ``<arrow_root>/<series>/latest.arrow`` through
        ``HistoricalData.get_arrow_file``, which slices to any date limits over
        the integer ``ts`` index and adds a ``datetime`` column. The frame is
        normalized to millisecond UTC and reduced to ``datetime`` plus the
        canonical measure columns, so the source switch is transparent to
        features, targets, and QC.
        """
        data = self._hd.get_arrow_file(
            file_path=str(arrow_latest_path(series)),
            start_date_limit=start_date_limit,
            end_date_limit=end_date_limit,
        )
        data = normalize_datetime(data)
        return data.select("datetime", *BAR_VALUE_COLUMNS)

    def get_time_bars(
        self,
        bar_size: str | int,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        row_count_limit: int | None = None,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        if row_count_limit is not None and n_rows is not None:
            raise ValueError("row_count_limit and n_rows are mutually exclusive.")
        start_date_limit, end_date_limit = _date_limits(
            start_date_limit, end_date_limit
        )
        name = time_bar_name(bar_size)
        series = time_bar_series(bar_size)
        limit = row_count_limit if row_count_limit is not None else n_rows
        criteria = {
            "kind": "arrow_bar_store",
            "bar": name,
            "series": series,
            "start": start_date_limit,
            "end": end_date_limit,
            "row_count_limit": row_count_limit,
            "n_rows": n_rows,
        }
        return self._cached(
            name,
            criteria,
            lambda: _limit_rows(
                self._load_arrow(series, start_date_limit, end_date_limit),
                limit,
            ),
            arrow_latest_path(series),
        )

    def get_dollar_bars(
        self,
        bar_size: str,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        row_count_limit: int | None = None,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        if row_count_limit is not None and n_rows is not None:
            raise ValueError("row_count_limit and n_rows are mutually exclusive.")
        start_date_limit, end_date_limit = _date_limits(
            start_date_limit, end_date_limit
        )
        name = dollar_bar_name(bar_size)
        series = dollar_bar_series(bar_size)
        limit = row_count_limit if row_count_limit is not None else n_rows
        criteria = {
            "kind": "arrow_bar_store",
            "bar": name,
            "series": series,
            "start": start_date_limit,
            "end": end_date_limit,
            "row_count_limit": row_count_limit,
            "n_rows": n_rows,
        }
        return self._cached(
            name,
            criteria,
            lambda: _limit_rows(
                self._load_arrow(series, start_date_limit, end_date_limit),
                limit,
            ),
            arrow_latest_path(series),
        )

    def _cached(
        self,
        label: str,
        criteria: dict[str, object],
        fetch: Callable[[], pl.DataFrame],
        source: Path,
    ) -> pl.DataFrame:
        path = cache_path(label, criteria)
        cached = read_fresh(path, source)
        if cached is not None:
            return self._store(cached)
        with _LIMEN_LOCK:
            cached = read_fresh(path, source)
            if cached is not None:
                return self._store(cached)
            data = fetch()
            write_cache(path, data, criteria)
            return self._store(data)

    def _store(self, data: pl.DataFrame) -> pl.DataFrame:
        self.data = data
        self.data_columns = data.columns
        return data
