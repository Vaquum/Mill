from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Callable

import polars as pl

import features
import limen.features as _LF
import limen.indicators as _LI

FactoryFn = Callable[..., pl.Expr]
Producer = Callable[..., pl.DataFrame]


def _raw_column(name: str) -> Callable[[], pl.Expr]:
    def factory() -> pl.Expr:
        return pl.col(name)
    return factory


REGISTRY: dict[str, dict[str, Any]] = {
    "open":             {"factory": _raw_column("open"),         "params": ()},
    "high":             {"factory": _raw_column("high"),         "params": ()},
    "low":              {"factory": _raw_column("low"),          "params": ()},
    "close":            {"factory": _raw_column("close"),        "params": ()},
    "volume":           {"factory": _raw_column("volume"),       "params": ()},
    "maker_ratio":      {"factory": _raw_column("maker_ratio"),  "params": ()},

    "roc": {
        "factory": features.roc_bps_per_bar,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "log_return": {
        "factory": features.log_return_bps_per_bar,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "realized_volatility": {
        "factory": features.realized_volatility_bps_per_bar,
        "params": ({"name": "n", "type": "int", "default": 16, "min": 1},),
    },
    "realized_semivolatility_down": {
        "factory": features.realized_semivolatility_down_bps_per_bar,
        "params": ({"name": "n", "type": "int", "default": 16, "min": 1},),
    },
    "parkinson_volatility": {
        "factory": features.parkinson_volatility_bps_per_bar,
        "params": ({"name": "n", "type": "int", "default": 4, "min": 1},),
    },
    "jump_volatility": {
        "factory": features.jump_volatility_bps_per_bar,
        "params": ({"name": "n", "type": "int", "default": 16, "min": 2},),
    },
    "atr": {
        "factory": features.atr_bps_per_bar,
        "params": ({"name": "n", "type": "int", "default": 14, "min": 1},),
    },
    "liquidity_z": {
        "factory": features.liquidity_z,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },
    "volume_z": {
        "factory": features.volume_z,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },
    "volatility_z": {
        "factory": features.volatility_z,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },
    "volume_to_liquidity_z": {
        "factory": features.volume_to_liquidity_z,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },
    "trades_z": {
        "factory": features.trades_z,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },
    "wick_proportion": {
        "factory": features.wick_proportion,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "stochastic_k_abs": {
        "factory": features.stochastic_k_abs,
        "params": ({"name": "n", "type": "int", "default": 14, "min": 2},),
    },
    "trade_size_ratio": {
        "factory": features.trade_size_ratio,
        "params": (
            {"name": "short_n", "type": "int", "default": 4, "min": 1},
            {"name": "long_n", "type": "int", "default": 96, "min": 2},
        ),
    },
    "volatility_ratio": {
        "factory": features.volatility_ratio,
        "params": (
            {"name": "short_n", "type": "int", "default": 4, "min": 1},
            {"name": "long_n", "type": "int", "default": 96, "min": 2},
        ),
    },
    "maker_ratio_trend": {
        "factory": features.maker_ratio_trend,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },
    "trade_imbalance": {
        "factory": features.trade_imbalance,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 1},),
    },
    "liquidity_range": {
        "factory": features.liquidity_range,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "trade_density": {
        "factory": features.trade_density,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "maker_volume_ratio": {
        "factory": features.maker_volume_ratio,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "close_position": {
        "factory": features.close_position,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "rsi": {
        "factory": features.rsi,
        "params": ({"name": "n", "type": "int", "default": 14, "min": 2},),
    },
    "distance_from_ma": {
        "factory": features.distance_from_ma,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 2},),
    },
    "close_ma_distance_atr": {
        "factory": features.close_ma_distance_atr,
        "params": (
            {"name": "n_ma",  "type": "int", "default": 20, "min": 1},
            {"name": "n_atr", "type": "int", "default": 14, "min": 1},
        ),
    },
    "high_low_range_trend": {
        "factory": features.high_low_range_trend,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },
    "kaufman_efficiency_ratio": {
        "factory": features.kaufman_efficiency_ratio,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 2},),
    },
    "return_autocorrelation": {
        "factory": features.return_autocorrelation,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 3},),
    },
    "volatility_of_volatility": {
        "factory": features.volatility_of_volatility,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 2},),
    },
    "volume_volatility_correlation": {
        "factory": features.volume_volatility_correlation,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 3},),
    },
    "return_volatility_correlation": {
        "factory": features.return_volatility_correlation,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 3},),
    },
    "downside_volatility_ratio": {
        "factory": features.downside_volatility_ratio,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 2},),
    },
    "roc_z": {
        "factory": features.roc_z,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 2},),
    },
    "maker_liquidity_share_z": {
        "factory": features.maker_liquidity_share_z,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 2},),
    },
    "taker_imbalance_z": {
        "factory": features.taker_imbalance_z,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 2},),
    },
    "taker_imbalance_abs_z": {
        "factory": features.taker_imbalance_abs_z,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 2},),
    },
    "taker_imbalance_ratio": {
        "factory": features.taker_imbalance_ratio,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 1},),
    },
    "narrow_range": {
        "factory": features.narrow_range,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 2},),
    },
    "volume_to_range": {
        "factory": features.volume_to_range,
        "params": ({"name": "n", "type": "int", "default": 24, "min": 1},),
    },
    "liquidity_drop": {
        "factory": features.liquidity_drop,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "volume_spike": {
        "factory": features.volume_spike,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "volatility_spike": {
        "factory": features.volatility_spike,
        "params": ({"name": "n", "type": "int", "default": 1, "min": 1},),
    },
    "abs_price_z": {
        "factory": features.abs_price_z,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },
    "trade_size_z": {
        "factory": features.trade_size_z,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },
    "maker_ratio_z": {
        "factory": features.maker_ratio_z,
        "params": ({"name": "n", "type": "int", "default": 96, "min": 2},),
    },

    "maker_volume_share_bps_per_bar":    {"factory": features.maker_volume_share_bps_per_bar,    "params": ()},
    "maker_liquidity_share_bps_per_bar": {"factory": features.maker_liquidity_share_bps_per_bar, "params": ()},

    "is_funding_hour":  {"factory": features.is_funding_hour,  "params": ()},
    "is_us_open_hour":  {"factory": features.is_us_open_hour,  "params": ()},
    "is_weekend":       {"factory": features.is_weekend,       "params": ()},

    "nth_minute":       {"factory": features.nth_minute,       "params": ()},
    "nth_hour":         {"factory": features.nth_hour,         "params": ()},
    "nth_day":          {"factory": features.nth_day,          "params": ()},
    "nth_month":        {"factory": features.nth_month,        "params": ()},
    "nth_quarter":      {"factory": features.nth_quarter,      "params": ()},
    "nth_half":         {"factory": features.nth_half,         "params": ()},
}


def _expr_producer(factory: FactoryFn) -> Producer:
    def run(raw: pl.DataFrame, **params: Any) -> pl.DataFrame:
        return raw.select(factory(**params) if params else factory())
    return run


def _limen_producer(fn: Callable[..., pl.DataFrame]) -> Producer:
    def run(raw: pl.DataFrame, **params: Any) -> pl.DataFrame:
        out = fn(raw, **params)
        new = [c for c in out.columns if c not in raw.columns]
        if not new:
            return out.select([])
        if not params:
            return out.select(new)
        suffix = "__" + "_".join(f"{k}={v}" for k, v in params.items())
        return out.select(new).rename({c: f"{c}{suffix}" for c in new})
    return run


def _limen_param_specs(fn: Callable[..., Any]) -> tuple[dict[str, Any], ...]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ()
    out: list[dict[str, Any]] = []
    for p in sig.parameters.values():
        if p.name in {"data", "df"}:
            continue
        default = p.default if p.default is not inspect.Parameter.empty else None
        if isinstance(default, bool):
            type_ = "bool"
        elif isinstance(default, int):
            type_ = "int"
        elif isinstance(default, float):
            type_ = "float"
        else:
            continue
        out.append({"name": p.name, "type": type_, "default": default})
    return tuple(out)


for _entry in REGISTRY.values():
    _entry["producer"] = _expr_producer(_entry["factory"])
    _entry["source"] = "local"


def _register_limen(ns: Any) -> None:
    for name in dir(ns):
        if name.startswith("_"):
            continue
        fn = getattr(ns, name)
        if not callable(fn):
            continue
        REGISTRY[name] = {
            "producer": _limen_producer(fn),
            "params": _limen_param_specs(fn),
            "source": "limen",
        }


_register_limen(_LF)
_register_limen(_LI)


def producer_for(factory_name: str) -> Producer:
    if factory_name not in REGISTRY:
        raise ValueError(f"Unknown factory: {factory_name!r}.")
    return REGISTRY[factory_name]["producer"]


def manifest() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "params": list(entry["params"]),
            "source": entry.get("source", "local"),
        }
        for name, entry in REGISTRY.items()
    ]
