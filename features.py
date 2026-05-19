from __future__ import annotations

import math

import polars as pl

_PARKINSON_SCALE = 4.0 * math.log(2.0)
_BIPOWER_SCALE = math.pi / 2.0


def roc_bps_per_bar(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    return (pl.col("close").pct_change(n) * (1e4 / n)).alias(f"roc_{n}_bps_per_bar")


def log_return_bps_per_bar(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    return (
        pl.col("close").pct_change(n).log1p() * (1e4 / n)
    ).alias(f"log_return_{n}_bps_per_bar")


def realized_volatility_bps_per_bar(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    sq = pl.col("close").log().diff().pow(2)
    return (
        (sq.rolling_sum(window_size=n) / n).sqrt() * 1e4
    ).alias(f"realized_volatility_{n}_bps_per_bar")


def realized_semivolatility_down_bps_per_bar(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    r = pl.col("close").log().diff()
    neg_sq = pl.when(r < 0).then(r.pow(2)).otherwise(0.0)
    return (
        (neg_sq.rolling_sum(window_size=n) / n).sqrt() * 1e4
    ).alias(f"realized_semivolatility_down_{n}_bps_per_bar")


def parkinson_volatility_bps_per_bar(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    pv = (pl.col("high").log() - pl.col("low").log()).pow(2) / _PARKINSON_SCALE
    return (
        (pv.rolling_sum(window_size=n) / n).sqrt() * 1e4
    ).alias(f"parkinson_volatility_{n}_bps_per_bar")


def jump_volatility_bps_per_bar(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    r = pl.col("close").log().diff()
    rv_pb = r.pow(2).rolling_sum(window_size=n) / n
    bv_pb = (
        _BIPOWER_SCALE
        * (r.abs() * r.abs().shift(1)).rolling_sum(window_size=n)
        / n
    )
    return (
        (rv_pb - bv_pb).clip(lower_bound=0.0).sqrt() * 1e4
    ).alias(f"jump_volatility_{n}_bps_per_bar")


def _true_range_price() -> pl.Expr:
    prev_close = pl.col("close").shift(1)
    return pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )


def atr_bps_per_bar(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    tr_bps = (_true_range_price() / pl.col("close").shift(1)) * 1e4
    return tr_bps.rolling_mean(window_size=n).alias(f"atr_{n}_bps_per_bar")


def liquidity_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    x = pl.col("liquidity_sum").log1p()
    mean = x.rolling_mean(window_size=n)
    std = x.rolling_std(window_size=n)
    return (
        pl.when(std == 0.0).then(None).otherwise((x - mean) / std)
    ).alias(f"liquidity_z_{n}")


def volume_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    x = pl.col("volume").log1p()
    mean = x.rolling_mean(window_size=n)
    std = x.rolling_std(window_size=n)
    return (
        pl.when(std == 0.0).then(None).otherwise((x - mean) / std)
    ).alias(f"volume_z_{n}")


def volatility_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    pv = (pl.col("high").log() - pl.col("low").log()).pow(2) / _PARKINSON_SCALE
    x = pv.log1p()
    mean = x.rolling_mean(window_size=n)
    std = x.rolling_std(window_size=n)
    return (
        pl.when(std == 0.0).then(None).otherwise((x - mean) / std)
    ).alias(f"volatility_z_{n}")


def trade_size_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    x = (pl.col("volume") / pl.col("no_of_trades").clip(lower_bound=1)).log1p()
    mean = x.rolling_mean(window_size=n)
    std = x.rolling_std(window_size=n)
    return (
        pl.when(std == 0.0).then(None).otherwise((x - mean) / std)
    ).alias(f"trade_size_z_{n}")


def volume_to_liquidity_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    x = (pl.col("volume") / pl.col("liquidity_sum")).log1p()
    mean = x.rolling_mean(window_size=n)
    std = x.rolling_std(window_size=n)
    return (
        pl.when(std == 0.0).then(None).otherwise((x - mean) / std)
    ).alias(f"volume_to_liquidity_z_{n}")

def trades_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    x = pl.col("no_of_trades").log1p()
    mean = x.rolling_mean(window_size=n)
    std = x.rolling_std(window_size=n)
    return (
        pl.when(std == 0.0).then(None).otherwise((x - mean) / std)
    ).alias(f"trades_z_{n}")

def wick_proportion(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low")
    prop = pl.when(rng == 0.0).then(0.0).otherwise((rng - body) / rng)
    return prop.rolling_mean(window_size=n).alias(f"wick_proportion_{n}")

def stochastic_k_abs(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    ll = pl.col("low").rolling_min(window_size=n)
    hh = pl.col("high").rolling_max(window_size=n)
    rng = hh - ll
    stoch = pl.when(rng == 0.0).then(0.5).otherwise((pl.col("close") - ll) / rng)
    return (stoch - 0.5).abs().alias(f"stochastic_k_abs_{n}")

def trade_size_ratio(short_n: int, long_n: int) -> pl.Expr:
    if short_n < 1 or long_n < short_n:
        raise ValueError("Invalid windows.")
    ts = pl.col("volume") / pl.col("no_of_trades").clip(lower_bound=1)
    short_mean = ts.rolling_mean(window_size=short_n)
    long_mean = ts.rolling_mean(window_size=long_n)
    return (short_mean / long_mean).alias(f"trade_size_ratio_{short_n}_{long_n}")

def volatility_ratio(short_n: int, long_n: int) -> pl.Expr:
    if short_n < 1 or long_n < short_n:
        raise ValueError("Invalid windows.")
    pv = (pl.col("high").log() - pl.col("low").log()).pow(2) / _PARKINSON_SCALE
    short_vol = pv.rolling_sum(window_size=short_n) / short_n
    long_vol = pv.rolling_sum(window_size=long_n) / long_n
    return pl.when(long_vol == 0.0).then(1.0).otherwise(short_vol / long_vol).alias(f"volatility_ratio_{short_n}_{long_n}")

def maker_ratio_trend(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    mr = pl.col("maker_ratio")
    mean = mr.rolling_mean(window_size=n)
    std = mr.rolling_std(window_size=n)
    return pl.when(std == 0.0).then(None).otherwise((mr - mean) / std).alias(f"maker_ratio_trend_{n}")

def trade_imbalance(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    # maker_volume is the volume that added liquidity.
    # We can look at maker_volume / volume over a window
    mv = pl.col("maker_volume").rolling_sum(window_size=n)
    tv = pl.col("volume").rolling_sum(window_size=n)
    return pl.when(tv == 0.0).then(0.5).otherwise(mv / tv).alias(f"trade_imbalance_{n}")

def liquidity_range(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    lr = pl.col("high_liquidity") / pl.col("low_liquidity").clip(lower_bound=1e-8)
    return lr.rolling_mean(window_size=n).alias(f"liquidity_range_{n}")

def trade_density(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    td = pl.col("no_of_trades") / pl.col("volume").clip(lower_bound=1e-8)
    return td.rolling_mean(window_size=n).alias(f"trade_density_{n}")

def maker_volume_ratio(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    mvr = pl.col("maker_volume") / pl.col("volume").clip(lower_bound=1e-8)
    return mvr.rolling_mean(window_size=n).alias(f"maker_volume_ratio_{n}")

def close_position(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    rng = pl.col("high") - pl.col("low")
    cp = pl.when(rng == 0.0).then(0.5).otherwise((pl.col("close") - pl.col("low")) / rng)
    return cp.rolling_mean(window_size=n).alias(f"close_position_{n}")

def rsi(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    delta = pl.col("close").diff()
    up = pl.when(delta > 0).then(delta).otherwise(0.0)
    down = pl.when(delta < 0).then(-delta).otherwise(0.0)
    
    # Simple moving average for RSI (Wilder used RMA, but SMA is fine for a feature)
    roll_up = up.rolling_mean(window_size=n)
    roll_down = down.rolling_mean(window_size=n)
    
    rs = roll_up / roll_down.clip(lower_bound=1e-8)
    return pl.when(roll_down == 0.0).then(100.0).otherwise(100.0 - (100.0 / (1.0 + rs))).alias(f"rsi_{n}")

def distance_from_ma(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    ma = pl.col("close").rolling_mean(window_size=n)
    return ((pl.col("close") - ma) / ma).alias(f"distance_from_ma_{n}")


def close_ma_distance_atr(n_ma: int, n_atr: int) -> pl.Expr:
    if n_ma < 1:
        raise ValueError("n_ma must be >= 1.")
    if n_atr < 1:
        raise ValueError("n_atr must be >= 1.")
    ma = pl.col("close").rolling_mean(window_size=n_ma)
    atr_price = _true_range_price().rolling_mean(window_size=n_atr)
    return (
        pl.when(atr_price == 0.0)
            .then(None)
            .otherwise((pl.col("close") - ma) / atr_price)
    ).alias(f"close_ma_distance_atr_{n_ma}_{n_atr}")

def high_low_range_trend(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    rng = pl.col("high") - pl.col("low")
    mean = rng.rolling_mean(window_size=n)
    std = rng.rolling_std(window_size=n)
    return pl.when(std == 0.0).then(None).otherwise((rng - mean) / std).alias(f"high_low_range_trend_{n}")

def kaufman_efficiency_ratio(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    direction = (pl.col("close") - pl.col("close").shift(n)).abs()
    volatility = (pl.col("close") - pl.col("close").shift(1)).abs().rolling_sum(window_size=n)
    return pl.when(volatility == 0.0).then(0.0).otherwise(direction / volatility).alias(f"kaufman_efficiency_ratio_{n}")

def return_autocorrelation(n: int) -> pl.Expr:
    if n < 3:
        raise ValueError("n must be >= 3.")
    ret = pl.col("close").pct_change()
    ret_shift = ret.shift(1)
    
    mean_ret = ret.rolling_mean(window_size=n)
    mean_ret_shift = ret_shift.rolling_mean(window_size=n)
    
    var_ret = ret.rolling_std(window_size=n)
    var_ret_shift = ret_shift.rolling_std(window_size=n)
    
    cov = ((ret - mean_ret) * (ret_shift - mean_ret_shift)).rolling_mean(window_size=n)
    
    denom = var_ret * var_ret_shift
    return pl.when(denom == 0.0).then(0.0).otherwise(cov / denom).alias(f"return_autocorrelation_{n}")

def volatility_of_volatility(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    pv = (pl.col("high").log() - pl.col("low").log()).pow(2) / _PARKINSON_SCALE
    return pv.rolling_std(window_size=n).alias(f"volatility_of_volatility_{n}")

def volume_volatility_correlation(n: int) -> pl.Expr:
    if n < 3:
        raise ValueError("n must be >= 3.")
    pv = (pl.col("high").log() - pl.col("low").log()).pow(2) / _PARKINSON_SCALE
    vol = pl.col("volume")
    
    mean_pv = pv.rolling_mean(window_size=n)
    mean_vol = vol.rolling_mean(window_size=n)
    
    var_pv = pv.rolling_std(window_size=n)
    var_vol = vol.rolling_std(window_size=n)
    
    cov = ((pv - mean_pv) * (vol - mean_vol)).rolling_mean(window_size=n)
    denom = var_pv * var_vol
    return pl.when(denom == 0.0).then(0.0).otherwise(cov / denom).alias(f"volume_volatility_correlation_{n}")

def return_volatility_correlation(n: int) -> pl.Expr:
    if n < 3:
        raise ValueError("n must be >= 3.")
    pv = (pl.col("high").log() - pl.col("low").log()).pow(2) / _PARKINSON_SCALE
    ret = pl.col("close").pct_change()
    
    mean_pv = pv.rolling_mean(window_size=n)
    mean_ret = ret.rolling_mean(window_size=n)
    
    var_pv = pv.rolling_std(window_size=n)
    var_ret = ret.rolling_std(window_size=n)
    
    cov = ((pv - mean_pv) * (ret - mean_ret)).rolling_mean(window_size=n)
    denom = var_pv * var_ret
    return pl.when(denom == 0.0).then(0.0).otherwise(cov / denom).alias(f"return_volatility_correlation_{n}")

def downside_volatility_ratio(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    r = pl.col("close").log().diff()
    sq = r.pow(2)
    neg_sq = pl.when(r < 0).then(sq).otherwise(0.0)
    
    rv = sq.rolling_sum(window_size=n)
    sv_down = neg_sq.rolling_sum(window_size=n)
    
    return pl.when(rv == 0.0).then(0.5).otherwise(sv_down / rv).alias(f"downside_volatility_ratio_{n}")

def roc_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    r = pl.col("close").pct_change()
    mean = r.rolling_mean(window_size=n)
    std = r.rolling_std(window_size=n)
    return pl.when(std == 0.0).then(None).otherwise((r - mean) / std).alias(f"roc_z_{n}")

def maker_liquidity_share_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    mls = pl.col("maker_liquidity") / pl.col("liquidity_sum").clip(lower_bound=1e-8)
    mean = mls.rolling_mean(window_size=n)
    std = mls.rolling_std(window_size=n)
    return pl.when(std == 0.0).then(None).otherwise((mls - mean) / std).alias(f"maker_liquidity_share_z_{n}")

def taker_imbalance_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    ti = pl.col("volume") * (1.0 - 2.0 * pl.col("maker_ratio"))
    mean = ti.rolling_mean(window_size=n)
    std = ti.rolling_std(window_size=n)
    return pl.when(std == 0.0).then(None).otherwise((ti - mean) / std).alias(f"taker_imbalance_z_{n}")

def taker_imbalance_abs_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    ti_abs = (pl.col("volume") * (1.0 - 2.0 * pl.col("maker_ratio"))).abs()
    mean = ti_abs.rolling_mean(window_size=n)
    std = ti_abs.rolling_std(window_size=n)
    return pl.when(std == 0.0).then(None).otherwise((ti_abs - mean) / std).alias(f"taker_imbalance_abs_z_{n}")

def taker_imbalance_ratio(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    ti_abs = (pl.col("volume") * (1.0 - 2.0 * pl.col("maker_ratio"))).abs()
    vol = pl.col("volume").clip(lower_bound=1e-8)
    return (ti_abs / vol).rolling_mean(window_size=n).alias(f"taker_imbalance_ratio_{n}")

def narrow_range(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    rng = pl.col("high") - pl.col("low")
    max_rng = rng.rolling_max(window_size=n)
    return pl.when(max_rng == 0.0).then(1.0).otherwise(rng / max_rng).alias(f"narrow_range_{n}")

def volume_to_range(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    rng = (pl.col("high") - pl.col("low")).clip(lower_bound=1e-8)
    vtr = pl.col("volume") / rng
    return vtr.rolling_mean(window_size=n).alias(f"volume_to_range_{n}")

def liquidity_drop(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    return (pl.col("liquidity_sum") / pl.col("liquidity_sum").shift(n).clip(lower_bound=1e-8)).alias(f"liquidity_drop_{n}")

def volume_spike(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    return (pl.col("volume") / pl.col("volume").shift(n).clip(lower_bound=1e-8)).alias(f"volume_spike_{n}")

def volatility_spike(n: int) -> pl.Expr:
    if n < 1:
        raise ValueError("n must be >= 1.")
    pv = (pl.col("high").log() - pl.col("low").log()).pow(2) / _PARKINSON_SCALE
    return (pv / pv.shift(n).clip(lower_bound=1e-8)).alias(f"volatility_spike_{n}")

def abs_price_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    close = pl.col("close")
    mean = close.rolling_mean(window_size=n)
    std = close.rolling_std(window_size=n)
    return pl.when(std == 0.0).then(None).otherwise(((close - mean) / std).abs()).alias(f"abs_price_z_{n}")

def maker_ratio_z(n: int) -> pl.Expr:
    if n < 2:
        raise ValueError("n must be >= 2.")
    x = pl.col("maker_ratio")
    mean = x.rolling_mean(window_size=n)
    std = x.rolling_std(window_size=n)
    return (
        pl.when(std == 0.0).then(None).otherwise((x - mean) / std)
    ).alias(f"maker_ratio_z_{n}")


def maker_volume_share_bps_per_bar() -> pl.Expr:
    return (
        (pl.col("maker_volume") / pl.col("volume")) * 1e4
    ).alias("maker_volume_share_bps_per_bar")


def maker_liquidity_share_bps_per_bar() -> pl.Expr:
    return (
        (pl.col("maker_liquidity") / pl.col("liquidity_sum")) * 1e4
    ).alias("maker_liquidity_share_bps_per_bar")


def is_funding_hour() -> pl.Expr:
    return (
        pl.col("datetime").dt.hour().is_in([0, 8, 16]).cast(pl.Int8)
    ).alias("is_funding_hour")


def is_us_open_hour() -> pl.Expr:
    return (
        (pl.col("datetime").dt.hour() == 14).cast(pl.Int8)
    ).alias("is_us_open_hour")


def is_weekend() -> pl.Expr:
    return (
        pl.col("datetime").dt.weekday().is_in([6, 7]).cast(pl.Int8)
    ).alias("is_weekend")


def nth_minute() -> pl.Expr:
    return (pl.col("datetime").dt.minute() + 1).alias("nth_minute")


def nth_hour() -> pl.Expr:
    return (pl.col("datetime").dt.hour() + 1).alias("nth_hour")


def nth_day() -> pl.Expr:
    return pl.col("datetime").dt.day().alias("nth_day")


def nth_month() -> pl.Expr:
    return pl.col("datetime").dt.month().alias("nth_month")


def nth_quarter() -> pl.Expr:
    return pl.col("datetime").dt.quarter().alias("nth_quarter")


def nth_half() -> pl.Expr:
    return ((pl.col("datetime").dt.month() - 1) // 6 + 1).alias("nth_half")
