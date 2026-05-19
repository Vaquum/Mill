from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.tsa.stattools import acf, adfuller, kpss


_LAGS = (1, 5, 24, 168)
_BLOCK_DEFAULT = 24
_BLOCK_THRESHOLD = 0.05
_BLOCK_MAX_LAG = 200
_ADF_KPSS_MAX_N = 5000
_BOOTSTRAP_ITER = 200
_BOOTSTRAP_MAX_N = 20000
_MI_MAX_N = 20000
_PAIRWISE_MAX_N = 20000
_PAIRWISE_MI_MAX_N = 5000
_PAIRWISE_MIN_N = 30
_RNG = np.random.default_rng(0)


def report(df: pl.DataFrame, target: str | None) -> dict:
    feature_cols = [c for c in df.columns if c not in ("datetime", target)]
    has_target = bool(target) and target in df.columns
    columns = list(feature_cols) + ([target] if has_target else [])

    arrays: dict[str, np.ndarray] = {
        col: df[col].cast(pl.Float64, strict=False).to_numpy() for col in columns
    }
    blocks = {col: _block_size(arrays[col]) for col in columns}

    rows: list[tuple[str, dict[str, object]]] = []
    rows.extend(_health_rows(df, arrays, columns))
    rows.append(("block_size", {col: blocks[col] for col in columns}))
    rows.extend(_pairwise_rows(arrays, feature_cols))

    if has_target:
        target_arr = arrays[target]
        rows.extend(_edge_rows(arrays, feature_cols, target_arr, blocks, target))
        rows.extend(_regime_rows(arrays, feature_cols, target_arr, target))

    return {
        "columns": ["metric", *columns],
        "rows": [[name, *(values.get(c) for c in columns)] for name, values in rows],
    }


def _health_rows(
    df: pl.DataFrame,
    arrays: dict[str, np.ndarray],
    columns: Sequence[str],
) -> list[tuple[str, dict[str, object]]]:
    n = df.height
    out: list[tuple[str, dict[str, object]]] = []

    out.append((
        "missing_rate",
        {c: float(df[c].null_count()) / n if n else None for c in columns},
    ))
    out.append((
        "unique_fraction",
        {c: float(df[c].n_unique()) / n if n else None for c in columns},
    ))
    for name, fn in (
        ("min", np.nanmin),
        ("max", np.nanmax),
        ("median", np.nanmedian),
        ("p01", lambda a: np.nanpercentile(a, 1)),
        ("p99", lambda a: np.nanpercentile(a, 99)),
        ("skew", lambda a: stats.skew(a, nan_policy="omit", bias=False)),
    ):
        out.append((name, {c: _safe(fn, _drop_nan(arrays[c])) for c in columns}))

    out.append(("adf_p", {c: _safe(_adf_p, _drop_nan(arrays[c])) for c in columns}))
    out.append(("kpss_p", {c: _safe(_kpss_p, _drop_nan(arrays[c])) for c in columns}))

    for lag in _LAGS:
        out.append((
            f"acf_lag_{lag}",
            {c: _acf_at(arrays[c], lag) for c in columns},
        ))

    return out


def _edge_rows(
    arrays: dict[str, np.ndarray],
    feature_cols: Sequence[str],
    target_arr: np.ndarray,
    blocks: dict[str, int],
    target_label: str,
) -> list[tuple[str, dict[str, object]]]:
    spearman: dict[str, object] = {}
    spear_lo: dict[str, object] = {}
    spear_hi: dict[str, object] = {}
    mi: dict[str, object] = {}
    spread: dict[str, object] = {}
    monotonic: dict[str, object] = {}

    for col in feature_cols:
        x = arrays[col]
        mask = ~np.isnan(x) & ~np.isnan(target_arr)
        xc, yc = x[mask], target_arr[mask]
        if xc.size < 100:
            spearman[col] = spear_lo[col] = spear_hi[col] = None
            mi[col] = spread[col] = monotonic[col] = None
            continue

        spearman[col] = _safe(lambda a: float(stats.spearmanr(a, yc).statistic), xc)
        lo, hi = _bootstrap_spearman_ci(xc, yc, blocks.get(col, _BLOCK_DEFAULT))
        spear_lo[col] = lo
        spear_hi[col] = hi
        mi[col] = _mutual_info(xc, yc)
        spread[col], monotonic[col] = _decile_metrics(xc, yc)

    rows: list[tuple[str, dict[str, object]]] = []
    for name, values in (
        ("spearman", spearman),
        ("spearman_ci_lo", spear_lo),
        ("spearman_ci_hi", spear_hi),
        ("mutual_information", mi),
        ("decile_spread", spread),
        ("monotonic_fraction", monotonic),
    ):
        values = dict(values)
        values[target_label] = None
        rows.append((name, values))
    return rows


def _regime_rows(
    arrays: dict[str, np.ndarray],
    feature_cols: Sequence[str],
    target_arr: np.ndarray,
    target_label: str,
) -> list[tuple[str, dict[str, object]]]:
    n = target_arr.size
    chunks = [(i * n // 4, (i + 1) * n // 4) for i in range(4)]
    agreement: dict[str, object] = {}
    spread: dict[str, object] = {}

    for col in feature_cols:
        x = arrays[col]
        values: list[float] = []
        for s, e in chunks:
            xc, yc = x[s:e], target_arr[s:e]
            mask = ~np.isnan(xc) & ~np.isnan(yc)
            if mask.sum() < 10:
                continue
            v = _safe(lambda a: float(stats.spearmanr(a, yc[mask]).statistic), xc[mask])
            if v is not None:
                values.append(v)
        if len(values) >= 2:
            arr = np.asarray(values)
            pos = int((arr > 0).sum())
            neg = int((arr < 0).sum())
            agreement[col] = f"{max(pos, neg)}/{len(values)}"
            spread[col] = float(np.std(arr))
        else:
            agreement[col] = spread[col] = None

    agreement[target_label] = None
    spread[target_label] = None
    return [
        ("regime_sign_agreement", agreement),
        ("regime_spearman_std", spread),
    ]


def _pairwise_rows(
    arrays: dict[str, np.ndarray],
    feature_cols: Sequence[str],
) -> list[tuple[str, dict[str, object]]]:
    n = len(feature_cols)
    empty = {c: None for c in feature_cols}
    if n < 2:
        return [
            ("max_pairwise_spearman", dict(empty)),
            ("mean_pairwise_spearman", dict(empty)),
            ("redundancy_penalty", dict(empty)),
            ("max_pairwise_mi", dict(empty)),
        ]

    mat = np.column_stack([arrays[c] for c in feature_cols])
    finite = ~np.isnan(mat).any(axis=1)
    sub = mat[finite]
    if sub.shape[0] < _PAIRWISE_MIN_N:
        return [
            ("max_pairwise_spearman", dict(empty)),
            ("mean_pairwise_spearman", dict(empty)),
            ("redundancy_penalty", dict(empty)),
            ("max_pairwise_mi", dict(empty)),
        ]

    if sub.shape[0] > _PAIRWISE_MAX_N:
        idx = np.linspace(0, sub.shape[0] - 1, _PAIRWISE_MAX_N).astype(np.int64)
        sub = sub[idx]

    rho = stats.spearmanr(sub, axis=0).statistic
    rho_arr = np.atleast_2d(np.asarray(rho, dtype=np.float64))
    if rho_arr.shape != (n, n):
        rho_arr = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    abs_rho = np.abs(rho_arr)
    np.fill_diagonal(abs_rho, np.nan)

    max_sp: dict[str, object] = {}
    mean_sp: dict[str, object] = {}
    redundancy: dict[str, object] = {}
    for i, c in enumerate(feature_cols):
        row = abs_rho[i]
        valid = row[~np.isnan(row)]
        if valid.size == 0:
            max_sp[c] = mean_sp[c] = redundancy[c] = None
        else:
            mx = float(np.max(valid))
            max_sp[c] = mx
            mean_sp[c] = float(np.mean(valid))
            redundancy[c] = float(1.0 - mx)

    if sub.shape[0] > _PAIRWISE_MI_MAX_N:
        idx = np.linspace(0, sub.shape[0] - 1, _PAIRWISE_MI_MAX_N).astype(np.int64)
        sub_mi = sub[idx]
    else:
        sub_mi = sub

    max_mi: dict[str, object] = {}
    for i in range(n):
        best: float | None = None
        xi = sub_mi[:, i:i + 1]
        for j in range(n):
            if i == j:
                continue
            try:
                v = float(mutual_info_regression(xi, sub_mi[:, j], random_state=0)[0])
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            if best is None or v > best:
                best = v
        max_mi[feature_cols[i]] = best

    return [
        ("max_pairwise_spearman", max_sp),
        ("mean_pairwise_spearman", mean_sp),
        ("redundancy_penalty", redundancy),
        ("max_pairwise_mi", max_mi),
    ]


def _drop_nan(arr: np.ndarray) -> np.ndarray:
    return arr[~np.isnan(arr)]


def _safe(fn, *args):
    try:
        result = fn(*args)
    except Exception:
        return None
    if isinstance(result, float) and (np.isnan(result) or np.isinf(result)):
        return None
    return result if result is not None else None


def _adf_p(arr: np.ndarray) -> float | None:
    if arr.size < 50:
        return None
    a = arr if arr.size <= _ADF_KPSS_MAX_N else _stride(arr, _ADF_KPSS_MAX_N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(a, maxlag=10, autolag=None)[1])


def _kpss_p(arr: np.ndarray) -> float | None:
    if arr.size < 50:
        return None
    a = arr if arr.size <= _ADF_KPSS_MAX_N else _stride(arr, _ADF_KPSS_MAX_N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(kpss(a, regression="c", nlags=10)[1])


def _stride(arr: np.ndarray, n: int) -> np.ndarray:
    step = max(1, arr.size // n)
    return arr[::step][:n]


def _acf_at(arr: np.ndarray, lag: int) -> float | None:
    a = _drop_nan(arr)
    if a.size <= lag + 1:
        return None
    try:
        coefs = acf(a, nlags=lag, fft=True)
        v = float(coefs[lag])
        return None if np.isnan(v) else v
    except Exception:
        return None


def _block_size(arr: np.ndarray) -> int:
    a = _drop_nan(arr)
    max_lag = min(_BLOCK_MAX_LAG, max(1, a.size - 2))
    if a.size <= 3:
        return _BLOCK_DEFAULT
    try:
        coefs = np.abs(acf(a, nlags=max_lag, fft=True))
    except Exception:
        return _BLOCK_DEFAULT
    for lag in range(1, len(coefs)):
        if coefs[lag] < _BLOCK_THRESHOLD:
            return int(lag)
    return _BLOCK_DEFAULT


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    block: int,
    n_iter: int = _BOOTSTRAP_ITER,
    ci: float = 0.95,
) -> tuple[float | None, float | None]:
    n = x.size
    if n > _BOOTSTRAP_MAX_N:
        idx = np.linspace(0, n - 1, _BOOTSTRAP_MAX_N).astype(np.int64)
        x = x[idx]
        y = y[idx]
        n = x.size
    if n < block + 1:
        return None, None
    starts = np.arange(n - block + 1)
    n_blocks = max(1, (n + block - 1) // block)
    estimates = np.empty(n_iter)
    for i in range(n_iter):
        s = _RNG.choice(starts, size=n_blocks, replace=True)
        idx = (s[:, None] + np.arange(block)).ravel()[:n]
        v = stats.spearmanr(x[idx], y[idx]).statistic
        estimates[i] = v if not np.isnan(v) else 0.0
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(estimates, 100 * alpha)),
        float(np.percentile(estimates, 100 * (1 - alpha))),
    )


def _mutual_info(x: np.ndarray, y: np.ndarray) -> float | None:
    n = x.size
    if n > _MI_MAX_N:
        idx = np.linspace(0, n - 1, _MI_MAX_N).astype(np.int64)
        x, y = x[idx], y[idx]
    try:
        if np.unique(y).size <= 10:
            mi = mutual_info_classif(
                x.reshape(-1, 1), y.astype(int), random_state=0
            )[0]
        else:
            mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0]
    except Exception:
        return None
    return float(mi)


def _decile_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    qs = np.quantile(x, np.linspace(0, 1, 11))
    if qs[0] == qs[-1]:
        return None, None
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    bins = np.digitize(x, qs[1:-1])
    means = np.empty(10)
    for k in range(10):
        sel = y[bins == k]
        means[k] = sel.mean() if sel.size else np.nan
    if np.any(np.isnan(means)):
        return None, None
    spread = float(means[-1] - means[0])
    diffs = np.diff(means)
    overall = 1 if spread >= 0 else -1
    same = int(np.sum(np.sign(diffs) == overall))
    return spread, same / 9.0
