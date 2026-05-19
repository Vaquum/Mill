from __future__ import annotations

from pathlib import Path

import polars as pl

from bars import (
    FIXED_DOLLAR_BAR_THRESHOLDS,
    dollar_bar_name,
    round_time_bars,
    time_bar_name,
    time_bar_seconds,
    time_bars_from_1m,
)
from cache_store import cache_path, read_fresh, write_cache
from dollar_bars import anchor_time_bar_name, dollar_bars_query, dollar_threshold
from frames import read_parquet, write_parquet
from remote_clickhouse import query_frame
from sources import resolve_limit, source_query


class GetData:
    def __init__(
        self,
        ssh_target: str = "root@37.27.112.167",
        database: str = "origo",
    ) -> None:
        self.ssh_target = ssh_target
        self.database = database
        self.data = pl.DataFrame()
        self.data_columns: list[str] = []

    def from_parquet(self, path: str | Path) -> pl.DataFrame:
        return self._store(read_parquet(path))

    def to_parquet(self, path: str | Path) -> Path:
        return write_parquet(self.data, path)

    def get_time_bars(
        self,
        bar_size: str | int,
        row_count_limit: int | None = None,
        random_row_count_limit: int | None = None,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        if (
            start_date_limit is not None
            and end_date_limit is not None
            and (
                row_count_limit is not None
                or random_row_count_limit is not None
                or n_rows is not None
            )
        ):
            raise ValueError(
                "row_count_limit, random_row_count_limit, and n_rows must be None "
                "when both start_date_limit and end_date_limit are set."
            )

        name = time_bar_name(bar_size)
        seconds = time_bar_seconds(name)
        limit, random_limit = resolve_limit(
            row_count_limit,
            random_row_count_limit,
            n_rows,
        )

        criteria = {
            "kind": "time_bars",
            "bar_name": name,
            "bar_seconds": seconds,
            "database": self.database,
            "start_date_limit": start_date_limit,
            "end_date_limit": end_date_limit,
            "row_count_limit": limit,
            "random_row_count_limit": None,
            "n_rows": None,
            "aggregation": "time_bars_from_1m_v1",
        }

        path = cache_path(name, criteria)
        if random_limit is None:
            cached = read_fresh(path)
            if cached is not None:
                return self._store(cached)

        data = self._time_bars(seconds, start_date_limit, end_date_limit)
        if random_limit is not None:
            return self._store(data.sample(n=random_limit, shuffle=True).sort("datetime"))
        if limit is not None:
            data = data.tail(limit)

        write_cache(path, data, criteria)
        return self._store(data)

    def get_spot_trades(
        self,
        row_count_limit: int | None = None,
        random_row_count_limit: int | None = None,
        kline_size: int = 60,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        return self._source(
            "spot_trades",
            row_count_limit,
            random_row_count_limit,
            kline_size,
            start_date_limit,
            end_date_limit,
            n_rows,
        )

    def get_dollar_bars(
        self,
        bar_size: str,
        row_count_limit: int | None = None,
        random_row_count_limit: int | None = None,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        if (
            start_date_limit is not None
            and end_date_limit is not None
            and (
                row_count_limit is not None
                or random_row_count_limit is not None
                or n_rows is not None
            )
        ):
            raise ValueError(
                "row_count_limit, random_row_count_limit, and n_rows must be None "
                "when both start_date_limit and end_date_limit are set."
            )

        name = dollar_bar_name(bar_size)
        limit, random_limit = resolve_limit(
            row_count_limit,
            random_row_count_limit,
            n_rows,
        )
        threshold = self._dollar_threshold(name, start_date_limit, end_date_limit)
        threshold_source = (
            "fixed_threshold_v1"
            if name in FIXED_DOLLAR_BAR_THRESHOLDS
            else "mean_anchor_time_bar_liquidity_sum_v1"
        )
        criteria = {
            "kind": "dollar_bars",
            "bar_name": name,
            "database": self.database,
            "start_date_limit": start_date_limit,
            "end_date_limit": end_date_limit,
            "row_count_limit": limit,
            "random_row_count_limit": None,
            "n_rows": None,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "aggregation": "dollar_bars_from_spot_trades_v1",
        }

        path = cache_path(name, criteria)
        if random_limit is None:
            cached = read_fresh(path)
            if cached is not None:
                return self._store(cached)

        data = self._dollar_bars(threshold, start_date_limit, end_date_limit)
        if random_limit is not None:
            return self._store(
                data.sample(n=random_limit, shuffle=True).sort("bar_start_datetime")
            )
        if limit is not None:
            data = data.tail(limit)

        write_cache(path, data, criteria)
        return self._store(data)

    def get_spot_klines(
        self,
        row_count_limit: int | None = None,
        random_row_count_limit: int | None = None,
        kline_size: int = 60,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        if kline_size == 60:
            return round_time_bars(
                self._source(
                    "spot_klines",
                    row_count_limit,
                    random_row_count_limit,
                    kline_size,
                    start_date_limit,
                    end_date_limit,
                    n_rows,
                )
            )
        return self.get_time_bars(
            kline_size,
            row_count_limit,
            random_row_count_limit,
            start_date_limit,
            end_date_limit,
            n_rows=n_rows,
        )

    def get_futures_trades(
        self,
        row_count_limit: int | None = None,
        random_row_count_limit: int | None = None,
        kline_size: int = 60,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        return self._source(
            "futures_trades",
            row_count_limit,
            random_row_count_limit,
            kline_size,
            start_date_limit,
            end_date_limit,
            n_rows,
        )

    def get_spot_book(
        self,
        row_count_limit: int | None = None,
        random_row_count_limit: int | None = None,
        kline_size: int = 60,
        start_date_limit: str | None = None,
        end_date_limit: str | None = None,
        *,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        return self._source(
            "spot_book",
            row_count_limit,
            random_row_count_limit,
            kline_size,
            start_date_limit,
            end_date_limit,
            n_rows,
        )

    def _time_bars(
        self,
        seconds: int,
        start_date_limit: str | None,
        end_date_limit: str | None,
    ) -> pl.DataFrame:
        base = self._one_minute_time_bars(start_date_limit, end_date_limit)
        return time_bars_from_1m(base, seconds)

    def _dollar_threshold(
        self,
        name: str,
        start_date_limit: str | None,
        end_date_limit: str | None,
    ) -> float:
        if name in FIXED_DOLLAR_BAR_THRESHOLDS:
            return dollar_threshold(FIXED_DOLLAR_BAR_THRESHOLDS[name])
        anchor = anchor_time_bar_name(name)
        time_bars = self.get_time_bars(
            anchor,
            start_date_limit=start_date_limit,
            end_date_limit=end_date_limit,
        )
        value = time_bars.select(pl.col("liquidity_sum").mean()).item()
        return dollar_threshold(float(value))

    def _dollar_bars(
        self,
        threshold: float,
        start_date_limit: str | None,
        end_date_limit: str | None,
    ) -> pl.DataFrame:
        query, parameters = dollar_bars_query(
            self.database,
            threshold,
            start_date_limit,
            end_date_limit,
        )
        return round_time_bars(query_frame(self.ssh_target, query, parameters))

    def _one_minute_time_bars(
        self,
        start_date_limit: str | None,
        end_date_limit: str | None,
    ) -> pl.DataFrame:
        criteria = {
            "kind": "time_bars_source",
            "bar_name": "1m_time_bars",
            "bar_seconds": 60,
            "database": self.database,
            "start_date_limit": start_date_limit,
            "end_date_limit": end_date_limit,
            "row_count_limit": None,
            "random_row_count_limit": None,
            "n_rows": None,
            "aggregation": "source_1m_v1",
        }
        path = cache_path("1m_time_bars", criteria)
        cached = read_fresh(path)
        if cached is not None:
            return cached

        data = round_time_bars(
            self._source(
                "spot_klines",
                None,
                None,
                60,
                start_date_limit,
                end_date_limit,
                None,
            )
        )
        write_cache(path, data, criteria)
        return data

    def _source(
        self,
        table_key: str,
        row_count_limit: int | None,
        random_row_count_limit: int | None,
        kline_size: int,
        start_date_limit: str | None,
        end_date_limit: str | None,
        n_rows: int | None,
    ) -> pl.DataFrame:
        if kline_size != 60:
            raise ValueError("Only kline_size=60 is supported for source tables.")
        query, parameters = source_query(
            table_key,
            self.database,
            row_count_limit,
            random_row_count_limit,
            start_date_limit,
            end_date_limit,
            n_rows,
        )
        return self._store(query_frame(self.ssh_target, query, parameters))

    def _store(self, data: pl.DataFrame) -> pl.DataFrame:
        self.data = data
        self.data_columns = data.columns
        return data
