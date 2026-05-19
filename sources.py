from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: tuple[str, ...]
    order_columns: tuple[str, ...]


TABLES = {
    "spot_trades": TableSpec(
        "binance_daily_spot_trades",
        (
            "datetime",
            "trade_id",
            "price",
            "quantity",
            "quote_quantity",
            "timestamp",
            "is_buyer_maker",
            "is_best_match",
        ),
        ("datetime", "trade_id"),
    ),
    "spot_klines": TableSpec(
        "binance_spot_klines",
        (
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "mean",
            "std",
            "volume",
            "maker_ratio",
            "no_of_trades",
            "open_liquidity",
            "high_liquidity",
            "low_liquidity",
            "close_liquidity",
            "liquidity_sum",
            "maker_volume",
            "maker_liquidity",
        ),
        ("datetime",),
    ),
    "futures_trades": TableSpec(
        "binance_daily_futures_trades",
        (
            "datetime",
            "futures_trade_id",
            "price",
            "quantity",
            "quote_quantity",
            "timestamp",
            "is_buyer_maker",
        ),
        ("datetime", "futures_trade_id"),
    ),
    "spot_book": TableSpec(
        "binance_spot_depth20_snapshots",
        ("datetime", "source_timestamp_ms", "last_update_id", "bids", "asks"),
        ("datetime", "source_timestamp_ms"),
    ),
}


def source_query(
    table_key: str,
    database: str,
    row_count_limit: int | None,
    random_row_count_limit: int | None,
    start_date_limit: str | None,
    end_date_limit: str | None,
    n_rows: int | None,
) -> tuple[str, dict[str, str | int]]:
    spec = TABLES[table_key]
    selected = ", ".join(spec.columns)
    where, parameters = where_sql(start_date_limit, end_date_limit)
    limit, random_limit = resolve_limit(row_count_limit, random_row_count_limit, n_rows)
    order_asc = order_sql(spec.order_columns, "ASC")

    if random_limit is not None:
        parameters["limit_rows"] = random_limit
        query = (
            f"SELECT {selected} FROM ("
            f"SELECT {selected} FROM {database}.{spec.name} {where} "
            "ORDER BY rand() LIMIT {limit_rows:UInt64}"
            f") ORDER BY {order_asc}"
        )
    elif limit is not None:
        parameters["limit_rows"] = limit
        order_desc = order_sql(spec.order_columns, "DESC")
        query = (
            f"SELECT {selected} FROM ("
            f"SELECT {selected} FROM {database}.{spec.name} {where} "
            f"ORDER BY {order_desc} LIMIT {{limit_rows:UInt64}}"
            f") ORDER BY {order_asc}"
        )
    else:
        query = f"SELECT {selected} FROM {database}.{spec.name} {where} ORDER BY {order_asc}"

    return query, parameters


def resolve_limit(
    row_count_limit: int | None,
    random_row_count_limit: int | None,
    n_rows: int | None,
) -> tuple[int | None, int | None]:
    if sum(v is not None for v in (row_count_limit, random_row_count_limit, n_rows)) > 1:
        raise ValueError(
            "Only one of row_count_limit, random_row_count_limit, and n_rows may be set."
        )

    limit = row_count_limit if row_count_limit is not None else n_rows
    random_limit = random_row_count_limit

    if limit is not None and limit < 1:
        raise ValueError("row_count_limit/n_rows must be positive.")
    if random_limit is not None and random_limit < 1:
        raise ValueError("random_row_count_limit must be positive.")

    return limit, random_limit


def where_sql(
    start_date_limit: str | None,
    end_date_limit: str | None,
) -> tuple[str, dict[str, str | int]]:
    parameters: dict[str, str | int] = {}
    clauses: list[str] = []

    if start_date_limit is not None:
        clauses.append("datetime >= parseDateTimeBestEffort({start_dt:String})")
        parameters["start_dt"] = start_date_limit.replace("T", " ")

    if end_date_limit is not None:
        clauses.append("datetime <= parseDateTimeBestEffort({end_dt:String})")
        parameters["end_dt"] = (
            f"{end_date_limit} 23:59:59"
            if len(end_date_limit) == 10
            else end_date_limit.replace("T", " ")
        )

    if not clauses:
        return "", parameters

    return "WHERE " + " AND ".join(clauses), parameters


def order_sql(columns: Sequence[str], direction: str) -> str:
    return ", ".join(f"{column} {direction}" for column in columns)
