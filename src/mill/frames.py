from __future__ import annotations

from pathlib import Path

import polars as pl


_EPOCH_SECONDS_LIMIT = 10_000_000_000


def normalize_datetime(data: pl.DataFrame) -> pl.DataFrame:
    if "datetime" not in data.columns:
        return data
    dtype = data.schema["datetime"]
    if dtype in (pl.UInt32, pl.UInt64, pl.Int32, pl.Int64):
        epoch = pl.col("datetime").cast(pl.Int64)
        return data.with_columns(
            pl.when(epoch.abs() < _EPOCH_SECONDS_LIMIT)
            .then(epoch * 1000)
            .otherwise(epoch)
            .cast(pl.Datetime("ms", time_zone="UTC"))
            .alias("datetime")
        )
    if dtype == pl.Date:
        return data.with_columns(
            pl.col("datetime")
            .cast(pl.Datetime("ms"))
            .dt.replace_time_zone("UTC")
            .alias("datetime")
        )
    if isinstance(dtype, pl.Datetime):
        expr = pl.col("datetime").dt.cast_time_unit("ms")
        expr = (
            expr.dt.replace_time_zone("UTC")
            if dtype.time_zone is None
            else expr.dt.convert_time_zone("UTC")
        )
        return data.with_columns(expr.alias("datetime"))
    return data


def read_parquet(path: str | Path) -> pl.DataFrame:
    return normalize_datetime(pl.read_parquet(path))


def write_parquet(data: pl.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(out)
    return out
