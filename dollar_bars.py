from __future__ import annotations

from bars import DOLLAR_BAR_ANCHORS
from sources import where_sql


DOLLAR_BAR_COLUMNS = (
    "bar_start_datetime",
    "bar_end_datetime",
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
)


def dollar_threshold(time_bars_liquidity_sum_mean: float) -> float:
    if time_bars_liquidity_sum_mean <= 0:
        raise ValueError("Dollar bar threshold must be positive.")
    return time_bars_liquidity_sum_mean


def anchor_time_bar_name(dollar_bar_name: str) -> str:
    if dollar_bar_name not in DOLLAR_BAR_ANCHORS:
        raise ValueError(f"Unsupported dollar bar name: {dollar_bar_name}.")
    return DOLLAR_BAR_ANCHORS[dollar_bar_name]


def dollar_bars_query(
    database: str,
    threshold: float,
    start_date_limit: str | None,
    end_date_limit: str | None,
) -> tuple[str, dict[str, str | int | float]]:
    where, source_parameters = where_sql(start_date_limit, end_date_limit)
    parameters: dict[str, str | int | float] = dict(source_parameters)
    parameters["threshold"] = threshold
    selected = ", ".join(DOLLAR_BAR_COLUMNS)
    query = (
        f"SELECT {selected} FROM ("
        "SELECT "
        "min(datetime) AS bar_start_datetime, "
        "max(datetime) AS bar_end_datetime, "
        "argMin(price, trade_id) AS open, "
        "max(price) AS high, "
        "min(price) AS low, "
        "argMax(price, trade_id) AS close, "
        "avg(price) AS mean, "
        "stddevPopStable(price) AS std, "
        "sumKahan(quantity) AS volume, "
        "avg(is_buyer_maker) AS maker_ratio, "
        "count() AS no_of_trades, "
        "argMin(price * quantity, trade_id) AS open_liquidity, "
        "max(price * quantity) AS high_liquidity, "
        "min(price * quantity) AS low_liquidity, "
        "argMax(price * quantity, trade_id) AS close_liquidity, "
        "sum(price * quantity) AS liquidity_sum, "
        "sumKahan(is_buyer_maker * quantity) AS maker_volume, "
        "sum(is_buyer_maker * price * quantity) AS maker_liquidity "
        "FROM ("
        "SELECT "
        "datetime, trade_id, price, quantity, is_buyer_maker, "
        "toUInt64(floor((cumulative_liquidity - trade_liquidity) / {threshold:Float64})) AS bar_id "
        "FROM ("
        "SELECT "
        "datetime, trade_id, price, quantity, is_buyer_maker, "
        "price * quantity AS trade_liquidity, "
        "sum(price * quantity) OVER (ORDER BY datetime, trade_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_liquidity "
        f"FROM {database}.binance_daily_spot_trades {where}"
        ")"
        ") GROUP BY bar_id "
        "ORDER BY bar_start_datetime ASC"
        ")"
    )
    return query, parameters
