from __future__ import annotations

from pathlib import Path

import polars as pl


def normalize_datetime(data: pl.DataFrame) -> pl.DataFrame:
    for column in ("datetime", "bar_start_datetime", "bar_end_datetime"):
        if column in data.columns:
            data = _normalize_datetime_column(data, column)
    return data


def _normalize_datetime_column(data: pl.DataFrame, column: str) -> pl.DataFrame:
    dtype = data.schema[column]
    if dtype in (pl.UInt32, pl.UInt64, pl.Int32, pl.Int64):
        return data.with_columns(
            (pl.col(column).cast(pl.Int64) * 1000)
            .cast(pl.Datetime("ms", time_zone="UTC"))
            .alias(column)
        )

    if isinstance(dtype, pl.Datetime):
        expr = pl.col(column).dt.cast_time_unit("ms")
        if dtype.time_zone is None:
            expr = expr.dt.replace_time_zone("UTC")
        else:
            expr = expr.dt.convert_time_zone("UTC")
        return data.with_columns(expr.alias(column))

    return data


def read_parquet(path: str | Path) -> pl.DataFrame:
    return normalize_datetime(pl.read_parquet(path))


def write_parquet(data: pl.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(out)
    return out
