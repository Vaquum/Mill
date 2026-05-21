from __future__ import annotations

from pathlib import Path

import polars as pl
from limen import HistoricalData

from bars import dollar_bar_usd, time_bar_seconds
from frames import read_parquet, write_parquet


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
        return self._store(
            self._hd.get_spot_klines(
                row_count_limit=row_count_limit,
                kline_size=time_bar_seconds(bar_size),
                start_date_limit=start_date_limit,
                end_date_limit=end_date_limit,
                n_rows=n_rows,
            )
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
        return self._store(
            self._hd.get_spot_dollar_klines(
                row_count_limit=row_count_limit,
                dollar_bar_size=dollar_bar_usd(bar_size),
                start_date_limit=start_date_limit,
                end_date_limit=end_date_limit,
                n_rows=n_rows,
            )
        )

    def _store(self, data: pl.DataFrame) -> pl.DataFrame:
        self.data = data
        self.data_columns = data.columns
        return data
