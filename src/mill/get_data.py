from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl
from limen import HistoricalData

from .bars import dollar_bar_name, dollar_bar_usd, time_bar_name, time_bar_seconds
from .cache_store import cache_path, read_fresh, write_cache
from .frames import read_parquet, write_parquet


_LIMEN_LOCK = threading.Lock()
_PROJECT_START = date(2024, 1, 1)
_PROJECT_END = date(2026, 5, 15)
_PROJECT_START_STR = _PROJECT_START.isoformat()
_PROJECT_END_STR = _PROJECT_END.isoformat()


def _date_part(value: str | None, default: date) -> date:
    if value is None:
        return default
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError as exc:
        raise ValueError(f"Invalid date limit: {value!r}.") from exc


def _bounded_dates(
    start_date_limit: str | None,
    end_date_limit: str | None,
) -> tuple[str, str]:
    start = _date_part(start_date_limit, _PROJECT_START)
    end = _date_part(end_date_limit, _PROJECT_END)
    if start < _PROJECT_START:
        raise ValueError(f"start_date_limit cannot be before {_PROJECT_START_STR}.")
    if end > _PROJECT_END:
        raise ValueError(f"end_date_limit cannot be after {_PROJECT_END_STR}.")
    if start > end:
        raise ValueError("start_date_limit cannot be after end_date_limit.")
    return start_date_limit or _PROJECT_START_STR, end_date_limit or _PROJECT_END_STR


def _project_window(data: pl.DataFrame) -> pl.DataFrame:
    if "datetime" not in data.columns:
        raise ValueError("Data must contain datetime to enforce project date bounds.")
    day = pl.col("datetime").dt.date()
    return data.filter((day >= pl.lit(_PROJECT_START)) & (day <= pl.lit(_PROJECT_END)))


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
        start_date_limit, end_date_limit = _bounded_dates(
            start_date_limit, end_date_limit
        )
        name = time_bar_name(bar_size)
        seconds = time_bar_seconds(bar_size)
        limit = row_count_limit if row_count_limit is not None else n_rows
        criteria = {
            "kind": "spot_klines",
            "bar": name,
            "kline_size": seconds,
            "start": start_date_limit,
            "end": end_date_limit,
            "row_count_limit": row_count_limit,
            "n_rows": n_rows,
        }
        return self._cached(
            name,
            criteria,
            lambda: _limit_rows(
                self._hd.get_spot_klines(
                    kline_size=seconds,
                    start_date_limit=start_date_limit,
                    end_date_limit=end_date_limit,
                ),
                limit,
            ),
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
        start_date_limit, end_date_limit = _bounded_dates(
            start_date_limit, end_date_limit
        )
        name = dollar_bar_name(bar_size)
        usd = dollar_bar_usd(bar_size)
        limit = row_count_limit if row_count_limit is not None else n_rows
        criteria = {
            "kind": "spot_dollar_klines",
            "bar": name,
            "dollar_bar_size": usd,
            "start": start_date_limit,
            "end": end_date_limit,
            "row_count_limit": row_count_limit,
            "n_rows": n_rows,
        }
        return self._cached(
            name,
            criteria,
            lambda: _limit_rows(
                self._hd.get_spot_dollar_klines(
                    dollar_bar_size=usd,
                    start_date_limit=start_date_limit,
                    end_date_limit=end_date_limit,
                ),
                limit,
            ),
        )

    def _cached(
        self,
        label: str,
        criteria: dict[str, object],
        fetch: Callable[[], pl.DataFrame],
    ) -> pl.DataFrame:
        path = cache_path(label, criteria)
        cached = read_fresh(path)
        if cached is not None:
            return self._store(cached)
        with _LIMEN_LOCK:
            cached = read_fresh(path)
            if cached is not None:
                return self._store(cached)
            data = _project_window(fetch())
            write_cache(path, data, criteria)
            return self._store(data)

    def _store(self, data: pl.DataFrame) -> pl.DataFrame:
        data = _project_window(data)
        self.data = data
        self.data_columns = data.columns
        return data
