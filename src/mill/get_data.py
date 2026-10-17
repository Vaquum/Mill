from __future__ import annotations

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
    time_bar_seconds,
    time_bar_series,
)
from .cache_store import cache_path, file_identity as source_identity, read_fresh, write_cache
from .frames import normalize_datetime, read_parquet, write_parquet
from .settings import arrow_latest_path, memory_budget
from .memo import ByteLRU, Flights


_DATA = ByteLRU(memory_budget("raw", 64), lambda value: value[1].estimated_size())
_DATA_FLIGHTS = Flights()


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


def _closed(data: pl.DataFrame, seconds: int, written_ns: int) -> pl.DataFrame:
    """The bars that had closed when the store wrote its file: it also writes the open bar, and rewrites it as trades arrive."""
    times = data["datetime"].dt.epoch("ms")
    if times.is_sorted():
        stop = times.search_sorted(written_ns // 1_000_000 - seconds * 1000, side="right")
        return data.slice(0, int(stop))
    return data.filter(pl.col("datetime").dt.epoch("ms") + seconds * 1000 <= written_ns // 1_000_000)


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
        source = arrow_latest_path(series)
        limit = row_count_limit if row_count_limit is not None else n_rows
        criteria = {
            "kind": "arrow_bar_store",
            "bar": name,
            "series": series,
            "start": start_date_limit,
            "end": end_date_limit,
            "row_count_limit": row_count_limit,
            "n_rows": n_rows,
            "closed": True,
        }

        def fetch() -> pl.DataFrame:
            # When the store wrote the file, read before the bars: a bar closed by then is closed in what is read.
            written = source.stat().st_mtime_ns
            bars = self._load_arrow(series, start_date_limit, end_date_limit)
            return _limit_rows(_closed(bars, time_bar_seconds(name), written), limit)

        return self._cached(name, criteria, fetch, source)

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
        key = str(path.resolve())
        identity = source_identity(source)
        hit = _DATA.get(key)
        if hit is not None and (identity is None or hit[0] == identity):
            return self._store(hit[1].clone())
        with _DATA_FLIGHTS.hold(key):
            identity = source_identity(source)
            hit = _DATA.get(key)
            if hit is not None and (identity is None or hit[0] == identity):
                return self._store(hit[1].clone())
            cached = read_fresh(path, source)
            if cached is not None:
                _DATA[key] = (identity, cached)
                return self._store(cached.clone())
            # The source's version before reading it: a write during the read leaves the cache stale.
            try:
                stamp = source.stat().st_mtime_ns
            except FileNotFoundError:
                stamp = None
            data = fetch()
            write_cache(path, data, criteria, stamp, identity)
            _DATA[key] = (identity, data)
            return self._store(data.clone())

    def _store(self, data: pl.DataFrame) -> pl.DataFrame:
        self.data = data
        self.data_columns = data.columns
        return data
