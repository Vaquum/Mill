from __future__ import annotations

import functools
import hashlib
import itertools
import warnings
import weakref
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

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


# ---- content-addressed memo for the expensive, pure QC kernels --------------
# Every per-column block (health/ADF/KPSS, block size), per-(feature, target)
# block (edge, regime), and per-pair kernel depends only on its input array
# bytes plus a few scalars, yet a /qc request recomputes all of them even when
# only the feature *set* changed. Memoizing them on a content hash of the
# inputs lets an incremental QC (add/remove a feature and re-run) reuse every
# unchanged feature's and pair's result. A cold first run is unchanged save a
# sub-millisecond hash per array. Results are bit-identical: a hit returns the
# exact value the function produced, a miss computes it exactly as before.
#
# The content hash itself is cached per array *object* (keyed on id, evicted
# via weakref.finalize when the array dies), so one report call hashes each
# column once, not once per kernel that sees it. Safe because no kernel — and
# nothing in report() — mutates an array after creating it.
#
# Thread note: serve runs QC on a ThreadingHTTPServer. Dict get/set are atomic
# under the GIL; the only non-atomic step is the size-bound clear, and every
# kernel here is deterministic, so a race can at worst recompute a value — never
# return a wrong one.
_KERNEL_CACHE: dict[object, object] = {}
_KERNEL_CACHE_MAX = 200_000
_MISS = object()
_DIGEST_CACHE: dict[int, tuple] = {}


def _arg_key(value: object) -> object:
    if isinstance(value, np.ndarray):
        key = _DIGEST_CACHE.get(id(value))
        if key is None:
            key = (
                "ndarray",
                value.shape,
                value.dtype.str,
                hashlib.blake2b(value.tobytes(), digest_size=16).digest(),
            )
            weakref.finalize(value, _DIGEST_CACHE.pop, id(value), None)
            _DIGEST_CACHE[id(value)] = key
        return key
    return value


def _memoized(fn):
    """Memoize a pure kernel on the content of its array args plus scalars."""
    name = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (
            name,
            tuple(_arg_key(a) for a in args),
            tuple(sorted((k, _arg_key(v)) for k, v in kwargs.items())),
        )
        cached = _KERNEL_CACHE.get(key, _MISS)
        if cached is not _MISS:
            return cached
        value = fn(*args, **kwargs)
        if len(_KERNEL_CACHE) >= _KERNEL_CACHE_MAX:
            _KERNEL_CACHE.clear()
        _KERNEL_CACHE[key] = value
        return value

    return wrapper


def clear_kernel_cache() -> None:
    """Drop all memoized kernel results (used by tests; harmless at runtime)."""
    _KERNEL_CACHE.clear()


def _resolve_block(b: object) -> int:
    return b if isinstance(b, int) and 1 <= b <= _BLOCK_MAX_LAG else _BLOCK_FALLBACK


def _seed_for(label: str) -> int:
    return zlib.crc32(label.encode()) & 0xFFFFFFFF


def _joint_block(blocks: dict[str, int | None], left: str, right: str) -> int:
    return max(_resolve_block(blocks.get(left)), _resolve_block(blocks.get(right)))


# ---- sweep: run the kernels once per pool, assemble reports per subset ------
# Nothing in a report couples features except the pairwise max/mean
# aggregation, so every value decomposes into per-column, per-(feature,
# target), and per-pair primitives that do not depend on which other features
# are present. precompute() runs the kernels once over a pool; report_for()
# assembles any subset's report from the maps without touching polars or the
# kernels, which is what makes exhaustive uni/pair/multi combination sweeps
# tractable: C(n, k) subsets cost C(n, 2) pair kernels, not C(n, k) reports.


@dataclass(frozen=True)
class QCPrimitives:
    """Subset-independent QC primitives for one feature pool.

    health is keyed by column (features plus target when present); edge and
    regime by feature; pairs by (left, right) in pool column order. The pair
    kernel is direction-sensitive (KNN mutual information), so report_for
    normalizes every subset to pool order to reuse exactly these values.
    """

    feature_cols: tuple[str, ...]
    target: str | None
    n_rows: int
    health: dict[str, tuple]
    pairs: dict[tuple[str, str], tuple]
    edge: dict[str, tuple]
    regime: dict[str, tuple]


def report(df: pl.DataFrame, target: str | None) -> dict:
    return report_for(precompute(df, target))


def precompute(df: pl.DataFrame, target: str | None) -> QCPrimitives:
    """Run every QC kernel once over the full feature pool of df."""
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

    if has_target:
        target_arr = arrays[target]
        edge = _edge_map(arrays, feature_cols, target_arr, blocks, target)
        regime = _regime_map(arrays, feature_cols, target_arr)
    else:
        edge = {}
        regime = {}

    return QCPrimitives(
        feature_cols=tuple(feature_cols),
        target=target if has_target else None,
        n_rows=df.height,
        health=_health_map(arrays, columns),
        pairs=_pair_map(arrays, feature_cols),
        edge=edge,
        regime=regime,
    )


def report_for(prim: QCPrimitives, cols: Sequence[str] | None = None) -> dict:
    """Assemble the report for a subset of the pool (default: all features).

    Bit-identical to report() on the frame restricted to cols: every value is
    a primitive lookup; only the pairwise aggregation depends on the subset.
    The target column is appended automatically and must not be requested.
    Columns follow pool order regardless of the order given.
    """
    subset = prim.feature_cols if cols is None else _pool_ordered(prim, cols)
    columns = list(subset) + ([prim.target] if prim.target else [])

    rows: list[tuple[str, dict[str, object]]] = []
    rows.extend(_health_rows(prim.health, columns, prim.n_rows))
    rows.extend(_pairwise_rows(prim.pairs, subset))
    if prim.target:
        rows.extend(_edge_rows(prim.edge, subset, prim.target))
        rows.extend(_regime_rows(prim.regime, subset, prim.target))

    return {
        "columns": ["metric", *columns],
        "rows": [[name, *(values.get(c) for c in columns)] for name, values in rows],
        "n_rows": prim.n_rows,
    }


def sweep(
    prim: QCPrimitives,
    sizes: Sequence[int],
) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Yield (cols, report) for every feature combination of each size.

    Pure aggregation over the precomputed primitives, so exhaustive
    enumeration is bounded by precompute(), not the combination count.
    """
    for k in sizes:
        for combo in itertools.combinations(prim.feature_cols, k):
            yield combo, report_for(prim, combo)


def _pool_ordered(prim: QCPrimitives, cols: Sequence[str]) -> tuple[str, ...]:
    requested = set(cols)
    unknown = requested.difference(prim.feature_cols)
    if unknown:
        raise ValueError(f"columns not in the precomputed pool: {sorted(unknown)}")
    return tuple(c for c in prim.feature_cols if c in requested)


def _health_map(
    arrays: dict[str, np.ndarray],
    columns: Sequence[str],
) -> dict[str, tuple]:
    return {c: _col_health(arrays[c]) for c in columns}


def _health_rows(
    health: dict[str, tuple],
    columns: Sequence[str],
    n: int,
) -> list[tuple[str, dict[str, object]]]:
    return [
        ("missing_rate", {c: float(health[c][0]) / n if n else None for c in columns}),
        ("n_unique", {c: health[c][1] for c in columns}),
        ("skew", {c: health[c][2] for c in columns}),
        ("adf_p", {c: health[c][3] for c in columns}),
        ("kpss_p", {c: health[c][4] for c in columns}),
    ]


@_memoized
def _col_health(arr: np.ndarray) -> tuple[object, ...]:
    """Whole per-column health block (missing/unique/skew/ADF/KPSS).

    Memoized as one kernel so a warm re-run pays one digest lookup per column
    instead of re-sorting it for n_unique and re-dropping NaNs four times.
    """
    finite = _drop_nan(arr)
    return (
        int(np.isnan(arr).sum()),
        int(np.unique(finite).size),
        _safe(lambda a: stats.skew(a, nan_policy="omit", bias=False), finite),
        _safe(_adf_p, finite),
        _safe(_kpss_p, finite),
    )


_EDGE_METRICS = (
    "spearman",
    "spearman_ci_lo",
    "spearman_ci_hi",
    "mutual_information",
    "decile_spread",
    "monotonic_fraction",
)


def _edge_map(
    arrays: dict[str, np.ndarray],
    feature_cols: Sequence[str],
    target_arr: np.ndarray,
    blocks: dict[str, int | None],
    target_label: str,
) -> dict[str, tuple]:
    return {
        col: _edge_metrics(
            arrays[col],
            target_arr,
            _joint_block(blocks, col, target_label),
            _seed_for(col),
        )
        for col in feature_cols
    }


def _edge_rows(
    edge: dict[str, tuple],
    feature_cols: Sequence[str],
    target_label: str,
) -> list[tuple[str, dict[str, object]]]:
    rows: list[tuple[str, dict[str, object]]] = []
    for idx, name in enumerate(_EDGE_METRICS):
        values: dict[str, object] = {col: edge[col][idx] for col in feature_cols}
        values[target_label] = None
        rows.append((name, values))
    return rows


@_memoized
def _edge_metrics(
    x: np.ndarray,
    y: np.ndarray,
    block: int,
    seed: int,
) -> tuple[object, ...]:
    """Whole per-feature edge block (mask, Spearman, CI, MI, deciles).

    Memoized as one kernel so a warm re-run is one digest lookup per feature
    instead of re-ranking full-length arrays for the point Spearman and
    deciles. The CI and MI beneath keep their own memos, so a cold call
    computes exactly what the inline loop did.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    xc, yc = x[mask], y[mask]
    if xc.size < 100:
        return None, None, None, None, None, None
    spearman = _safe(lambda a: float(stats.spearmanr(a, yc).statistic), xc)
    lo, hi = _bootstrap_spearman_ci(xc, yc, block, seed=seed)
    mi = _mutual_info(xc, yc)
    spread, monotonic = _decile_metrics(xc, yc)
    return spearman, lo, hi, mi, spread, monotonic


def _regime_map(
    arrays: dict[str, np.ndarray],
    feature_cols: Sequence[str],
    target_arr: np.ndarray,
) -> dict[str, tuple]:
    return {col: _regime_metrics(arrays[col], target_arr) for col in feature_cols}


def _regime_rows(
    regime: dict[str, tuple],
    feature_cols: Sequence[str],
    target_label: str,
) -> list[tuple[str, dict[str, object]]]:
    agreement: dict[str, object] = {col: regime[col][0] for col in feature_cols}
    spread: dict[str, object] = {col: regime[col][1] for col in feature_cols}
    agreement[target_label] = None
    spread[target_label] = None
    return [
        ("regime_sign_agreement", agreement),
        ("regime_spearman_std", spread),
    ]


@_memoized
def _regime_metrics(x: np.ndarray, y: np.ndarray) -> tuple[object, object]:
    """Per-feature regime block over four equal time chunks, memoized whole:
    the four chunked Spearmans were the only kernels a warm re-run still paid.
    """
    n = y.size
    chunks = [(i * n // 4, (i + 1) * n // 4) for i in range(4)]
    values: list[float] = []
    for s, e in chunks:
        xc, yc = x[s:e], y[s:e]
        mask = ~np.isnan(xc) & ~np.isnan(yc)
        if mask.sum() < 10:
            continue
        v = _safe(lambda a: float(stats.spearmanr(a, yc[mask]).statistic), xc[mask])
        if v is not None:
            values.append(v)
    if len(values) < 2:
        return None, None
    arr = np.asarray(values)
    pos = int((arr > 0).sum())
    neg = int((arr < 0).sum())
    return f"{max(pos, neg)}/{len(values)}", float(np.std(arr))


def _pair_map(
    arrays: dict[str, np.ndarray],
    feature_cols: Sequence[str],
) -> dict[tuple[str, str], tuple]:
    pairs: dict[tuple[str, str], tuple] = {}
    for i, left in enumerate(feature_cols):
        xi = arrays[left]
        for right in feature_cols[i + 1 :]:
            pairs[(left, right)] = _pair_metrics(xi, arrays[right])
    return pairs


def _pairwise_rows(
    pairs: dict[tuple[str, str], tuple],
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
        for j in range(i + 1, n):
            abs_r, mi = pairs[(feature_cols[i], feature_cols[j])]
            if abs_r is not None:
                abs_rho[i, j] = abs_rho[j, i] = abs_r
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


@_memoized
def _pair_metrics(
    xi: np.ndarray, xj: np.ndarray
) -> tuple[float | None, float | None]:
    """Absolute Spearman and mutual information for one feature pair.

    Extracted verbatim from the pairwise loop so the whole per-pair cost (mask,
    downsample, Spearman, MI) is memoized on the pair's content: an incremental
    QC then recomputes only the pairs that involve a changed feature, not all
    O(k^2). Keyed by (xi, xj) in the loop's i<j order, so the value matches what
    the inline loop produced.
    """
    mask = ~np.isnan(xi) & ~np.isnan(xj)
    if int(mask.sum()) < _PAIRWISE_MIN_N:
        return None, None
    left = xi[mask]
    right = xj[mask]
    sp_left, sp_right = _downsample_pair(left, right, _PAIRWISE_MAX_N)
    rho = _safe(
        lambda a, b: float(stats.spearmanr(a, b).statistic),
        sp_left,
        sp_right,
    )
    abs_rho = abs(float(rho)) if rho is not None else None
    mi_left, mi_right = _downsample_pair(left, right, _PAIRWISE_MI_MAX_N)
    mi = _pairwise_mi(mi_left, mi_right)
    return abs_rho, mi


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


@_memoized
def _adf_p(arr: np.ndarray) -> float | None:
    if arr.size < 50:
        return None
    a = arr if arr.size <= _ADF_KPSS_MAX_N else _stride(arr, _ADF_KPSS_MAX_N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(a, maxlag=10, autolag=None)[1])


@_memoized
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


@_memoized
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


@_memoized
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


@_memoized
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
