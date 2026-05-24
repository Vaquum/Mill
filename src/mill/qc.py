from __future__ import annotations

import warnings
import zlib
from collections.abc import Sequence

import numpy as np
import polars as pl
from numba import njit, prange
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.tsa.stattools import acf, adfuller, kpss


_BLOCK_FALLBACK = 24
_BLOCK_THRESHOLD = 0.05
_BLOCK_MAX_LAG = 200
_ADF_KPSS_MAX_N = 5000
_BOOTSTRAP_ITER = 200
_BOOTSTRAP_MAX_N = 20000
_MI_MAX_N = 20000
_PAIRWISE_MAX_N = 20000
_PAIRWISE_MI_MAX_N = 5000
_PAIRWISE_MIN_N = 30


def _resolve_block(b: object) -> int:
    return b if isinstance(b, int) and 1 <= b <= _BLOCK_MAX_LAG else _BLOCK_FALLBACK


def _seed_for(label: str) -> int:
    return zlib.crc32(label.encode()) & 0xFFFFFFFF


def _joint_block(blocks: dict[str, int | None], left: str, right: str) -> int:
    return max(_resolve_block(blocks.get(left)), _resolve_block(blocks.get(right)))


def report(df: pl.DataFrame, target: str | None) -> dict:
    if "datetime" in df.columns:
        df = df.sort("datetime")
    skip = {"datetime"} | ({target} if target else set())
    feature_cols = [c for c in df.columns if c not in skip]
    has_target = bool(target) and target in df.columns
    columns = list(feature_cols) + ([target] if has_target else [])

    arrays: dict[str, np.ndarray] = {
        col: df[col].cast(pl.Float64, strict=False).to_numpy() for col in columns
    }
    blocks: dict[str, int | None] = {col: _block_size(arrays[col]) for col in columns}

    rows: list[tuple[str, dict[str, object]]] = []
    rows.extend(_health_rows(df, arrays, columns))
    rows.extend(_pairwise_rows(arrays, feature_cols))

    if has_target:
        target_arr = arrays[target]
        rows.extend(_edge_rows(arrays, feature_cols, target_arr, blocks, target))
        rows.extend(_regime_rows(arrays, feature_cols, target_arr, target))

    return {
        "columns": ["metric", *columns],
        "rows": [[name, *(values.get(c) for c in columns)] for name, values in rows],
        "n_rows": df.height,
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
        {c: float(np.isnan(arrays[c]).sum()) / n if n else None for c in columns},
    ))
    out.append((
        "n_unique",
        {c: int(np.unique(_drop_nan(arrays[c])).size) for c in columns},
    ))
    out.append((
        "skew",
        {c: _safe(lambda a: stats.skew(a, nan_policy="omit", bias=False), _drop_nan(arrays[c])) for c in columns},
    ))

    out.append(("adf_p", {c: _safe(_adf_p, _drop_nan(arrays[c])) for c in columns}))
    out.append(("kpss_p", {c: _safe(_kpss_p, _drop_nan(arrays[c])) for c in columns}))

    return out


def _edge_rows(
    arrays: dict[str, np.ndarray],
    feature_cols: Sequence[str],
    target_arr: np.ndarray,
    blocks: dict[str, int | None],
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
        lo, hi = _bootstrap_spearman_ci(
            xc, yc, _joint_block(blocks, col, target_label), seed=_seed_for(col),
        )
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

    abs_rho = np.full((n, n), np.nan, dtype=np.float64)
    max_mi_values: list[float | None] = [None] * n

    for i in range(n):
        xi = arrays[feature_cols[i]]
        for j in range(i + 1, n):
            xj = arrays[feature_cols[j]]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if int(mask.sum()) < _PAIRWISE_MIN_N:
                continue
            left = xi[mask]
            right = xj[mask]
            sp_left, sp_right = _downsample_pair(left, right, _PAIRWISE_MAX_N)
            rho = _safe(
                lambda a, b: float(stats.spearmanr(a, b).statistic),
                sp_left,
                sp_right,
            )
            if rho is not None:
                abs_rho[i, j] = abs_rho[j, i] = abs(float(rho))
            mi_left, mi_right = _downsample_pair(left, right, _PAIRWISE_MI_MAX_N)
            mi = _pairwise_mi(mi_left, mi_right)
            if mi is not None:
                max_mi_values[i] = (
                    mi if max_mi_values[i] is None else max(max_mi_values[i], mi)
                )
                max_mi_values[j] = (
                    mi if max_mi_values[j] is None else max(max_mi_values[j], mi)
                )

    max_sp: dict[str, object] = {}
    mean_sp: dict[str, object] = {}
    redundancy: dict[str, object] = {}
    max_mi: dict[str, object] = {}
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
        max_mi[c] = max_mi_values[i]

    return [
        ("max_pairwise_spearman", max_sp),
        ("mean_pairwise_spearman", mean_sp),
        ("redundancy_penalty", redundancy),
        ("max_pairwise_mi", max_mi),
    ]


def _downsample_pair(
    left: np.ndarray,
    right: np.ndarray,
    max_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    if left.size <= max_n:
        return left, right
    idx = np.linspace(0, left.size - 1, max_n).astype(np.int64)
    return left[idx], right[idx]


def _pairwise_mi(left: np.ndarray, right: np.ndarray) -> float | None:
    try:
        value = mutual_info_regression(
            left.reshape(-1, 1), right, random_state=0
        )[0]
    except Exception:
        return None
    return float(value) if np.isfinite(value) else None


def _drop_nan(arr: np.ndarray) -> np.ndarray:
    return arr[~np.isnan(arr)]


def _safe(fn, *args):
    try:
        result = fn(*args)
    except Exception:
        return None
    if result is None:
        return None
    if isinstance(result, np.generic):
        result = result.item()
    if isinstance(result, float) and not np.isfinite(result):
        return None
    return result


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


def _block_size(arr: np.ndarray) -> int | None:
    a = _drop_nan(arr)
    if a.size <= 3:
        return None
    max_lag = min(_BLOCK_MAX_LAG, max(1, a.size - 2))
    try:
        coefs = np.abs(acf(a, nlags=max_lag, fft=True))
    except Exception:
        return None
    for lag in range(1, len(coefs)):
        if coefs[lag] < _BLOCK_THRESHOLD:
            return int(lag)
    return _BLOCK_MAX_LAG


@njit
def _rankdata_1d(arr: np.ndarray) -> np.ndarray:
    n = arr.size
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


@njit
def _pearson(arr_x: np.ndarray, arr_y: np.ndarray) -> float:
    arr_x = arr_x - arr_x.mean()
    arr_y = arr_y - arr_y.mean()
    denom = np.sqrt((arr_x * arr_x).sum() * (arr_y * arr_y).sum())
    if denom == 0.0:
        return 0.0
    return float((arr_x * arr_y).sum() / denom)


@njit(parallel=True)
def _bootstrap_spearman_estimates(
    arr_x: np.ndarray,
    arr_y: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    n_iter = idx.shape[0]
    estimates = np.empty(n_iter, dtype=np.float64)
    for i in prange(n_iter):
        sample_x = arr_x[idx[i]]
        sample_y = arr_y[idx[i]]
        value = _pearson(_rankdata_1d(sample_x), _rankdata_1d(sample_y))
        estimates[i] = 0.0 if np.isnan(value) else value
    return estimates


def _bootstrap_block_indices(
    n: int,
    block: int,
    n_iter: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    n_blocks = max(1, (n + block - 1) // block)
    block_starts = rng.choice(starts, size=(n_iter, n_blocks), replace=True)
    return (
        block_starts[:, :, None] + np.arange(block)
    ).reshape(n_iter, -1)[:, :n]


def _warmup_bootstrap() -> None:
    arr = np.arange(32, dtype=np.float64)
    idx = np.arange(32, dtype=np.int64).reshape(1, -1)
    _bootstrap_spearman_estimates(arr, arr, idx)


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    block: int,
    *,
    seed: int = 0,
    n_iter: int = _BOOTSTRAP_ITER,
    ci: float = 0.95,
) -> tuple[float | None, float | None]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = x.size
    if n > _BOOTSTRAP_MAX_N:
        idx = np.linspace(0, n - 1, _BOOTSTRAP_MAX_N).astype(np.int64)
        x = x[idx]
        y = y[idx]
        n = x.size
    if n < block + 1:
        return None, None
    idx = _bootstrap_block_indices(n, block, n_iter, seed)
    estimates = _bootstrap_spearman_estimates(x, y, idx)
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
        values = np.unique(y)
        if values.size < 2:
            return 0.0
        if values.size <= 10 and np.all(values == np.round(values)):
            labels = np.unique(y, return_inverse=True)[1]
            mi = mutual_info_classif(
                x.reshape(-1, 1), labels, random_state=0
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


_warmup_bootstrap()
