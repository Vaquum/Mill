from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import polars as pl
from limen import HistoricalData

from bars import dollar_bar_name, dollar_bar_usd, time_bar_name, time_bar_seconds
from cache_store import cache_path, read_fresh, write_cache
from frames import read_parquet, write_parquet


_LIMEN_LOCK = threading.Lock()


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
        name = time_bar_name(bar_size)
        seconds = time_bar_seconds(bar_size)
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
            lambda: self._hd.get_spot_klines(
                row_count_limit=row_count_limit,
                kline_size=seconds,
                start_date_limit=start_date_limit,
                end_date_limit=end_date_limit,
                n_rows=n_rows,
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
        name = dollar_bar_name(bar_size)
        usd = dollar_bar_usd(bar_size)
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
            lambda: self._hd.get_spot_dollar_klines(
                row_count_limit=row_count_limit,
                dollar_bar_size=usd,
                start_date_limit=start_date_limit,
                end_date_limit=end_date_limit,
                n_rows=n_rows,
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
            data = fetch()
            write_cache(path, data, criteria)
            return self._store(data)

    def _store(self, data: pl.DataFrame) -> pl.DataFrame:
        self.data = data
        self.data_columns = data.columns
        return data
