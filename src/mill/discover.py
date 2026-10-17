"""Find features: which pool columns describe marked market states, and is that chance?

Stage one ranks every column of a feature pool by the AUC of its values between
the marked bars and the rest of a context, from per-column ranks and one matrix
product. The chance check is an exact circular-shift null of the best separation
in the whole pool: every shift that moves each marked run off itself, computed
for all columns at once by FFT. A column reads "beyond chance" when its adjusted
p, the share of labelings (the observed one counted in) whose best column
separates at least as well, is at most 1 in 20: family-wise across the pool.
Stage two keeps one leader per correlation cluster, adds moving-block bootstrap
intervals, and fits a Limen threshold rule with persistence to each leader.

Nothing here creates a feature: it evaluates the registry's producers on Mill's
own bars, as the store holds them through the latest bar.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
from typing import Any

import numpy as np
import polars as pl
from limen.sfd.rule_based.predicates import build_predicate

from . import feature_registry

MODES = ("inside", "lead")
MAX_LEADERS = 12
MIN_EVENTS_FOR_INTERVALS = 30
_SCREEN_LEADERS = 40
_CLUSTER_RHO = 0.9
_BOOTSTRAP_DRAWS = 200
_MIN_BLOCK = 24
_PERSISTENCE = (1, 2, 3)
_FFT_CHUNK = 64
_BEYOND_ONE_IN = 20
# FFT and direct sums agree far closer than this; a near tie counts against the column.
_TIE = 1e-9


@dataclass(frozen=True)
class Pool:
    """A feature pool on one bar series: every column, where it came from, its values."""

    names: tuple[str, ...]
    natives: tuple[str, ...]
    factories: tuple[str, ...]
    sources: tuple[str, ...]
    params: tuple[dict[str, Any], ...]
    ms: np.ndarray
    values: np.ndarray
    skipped: tuple[str, ...]


def catalog_pool(raw: pl.DataFrame, pause: Callable[[], None] | None = None) -> Pool:
    """Every registry producer whose parameters all have defaults, at those defaults."""
    entries = []
    for factory, entry in feature_registry.REGISTRY.items():
        params = {p["name"]: p["default"] for p in entry["params"]}
        if all(value is not None for value in params.values()):
            entries.append((factory, entry["source"], params, entry["producer"], {}))
    return _build(raw, entries, pause)


def spec_pool(raw: pl.DataFrame, spec: Sequence[dict[str, Any]], pause: Callable[[], None] | None = None) -> Pool:
    """The columns of a Mill view's spec, each with the parameters the view gives it."""
    entries = []
    for item in spec:
        factory, given = item["factory"], dict(item.get("params") or {})
        entry = feature_registry.REGISTRY[factory]
        params = {p["name"]: p["default"] for p in entry["params"]} | given
        entries.append((factory, entry["source"], params, entry["producer"], given))
    return _build(raw, entries, pause)


def _build(raw: pl.DataFrame, entries: list, pause: Callable[[], None] | None = None) -> Pool:
    """Each entry's columns; pause, when given, runs before each producer, so a caller can hold the build there."""
    names: list[str] = []
    natives: list[str] = []
    meta: list[tuple[str, str, dict[str, Any]]] = []
    arrays: list[np.ndarray] = []
    skipped: list[str] = []
    for factory, source, params, producer, given in entries:
        if pause is not None:
            pause()
        try:
            part = producer(raw, **given)
        except Exception:
            skipped.append(factory)
            continue
        for column in part.columns:
            dtype = part.schema[column]
            if not (dtype.is_numeric() or dtype == pl.Boolean):
                continue
            names.append(column if column not in names else f"{factory}:{column}")
            # Mill suffixes a column with the parameters a view gives it; Limen names it without.
            natives.append(column.split("__", 1)[0] if given else column)
            meta.append((factory, source, params))
            arrays.append(part[column].cast(pl.Float64).to_numpy())
    values = np.column_stack(arrays) if arrays else np.empty((raw.height, 0))
    values[~np.isfinite(values)] = np.nan
    return Pool(
        names=tuple(names),
        natives=tuple(natives),
        factories=tuple(m[0] for m in meta),
        sources=tuple(m[1] for m in meta),
        params=tuple(m[2] for m in meta),
        ms=raw["datetime"].dt.epoch("ms").to_numpy().astype(np.int64),
        values=values,
        skipped=tuple(skipped),
    )


def parse_ms(value: Any) -> int:
    """An ISO-8601 instant (UTC when no zone is given) or integer milliseconds."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"invalid instant: {value!r}")
    if isinstance(value, int):
        return value
    moment = datetime.fromisoformat(value.strip())
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


def check_span(start_ms: int, end_ms: int) -> None:
    """Refuse an empty span; a span without bars simply marks none."""
    if start_ms >= end_ms:
        raise ValueError("a window must end after it starts")


def day_of(ms: int) -> date:
    """The UTC day an instant falls on."""
    return datetime.fromtimestamp(ms / 1000, timezone.utc).date()


def overlap(ms: np.ndarray, step_ms: int, start_ms: int, end_ms: int) -> np.ndarray:
    """Bars whose span [open, open + step) meets [start, end)."""
    return (ms < end_ms) & (ms + step_ms > start_ms)


def target(
    ms: np.ndarray, step_ms: int, windows: Sequence[tuple[int, int]], mode: str, lead_bars: int
) -> tuple[np.ndarray, np.ndarray]:
    """Marked bars, and bars that leave the comparison (lead mode leaves the events out).

    inside: every bar that meets a window. lead: the lead_bars bars before each
    window's first bar; the window's own bars are neither marked nor compared.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    marked = np.zeros(ms.size, dtype=bool)
    left_out = np.zeros(ms.size, dtype=bool)
    for start_ms, end_ms in windows:
        inside = overlap(ms, step_ms, start_ms, end_ms)
        if mode == "inside":
            marked |= inside
            continue
        hits = np.flatnonzero(inside)
        if hits.size:
            first = int(hits[0])
            marked[max(0, first - lead_bars):first] = True
            left_out |= inside
    return marked & ~left_out, left_out


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as [start, stop) index pairs."""
    edges = np.diff(np.r_[0, flags.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def column_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks of each column among its finite rows; NaN where undefined."""
    if values.shape[1] == 0:
        return values.copy()
    frame = pl.from_numpy(values).fill_nan(None)
    return frame.select(pl.all().rank("average")).to_numpy().astype(np.float64)


def auc(y: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """AUC of every column between marked (y = 1) and other rows, from ranks."""
    valid = ~np.isnan(ranks)
    n1 = y @ valid
    n0 = (1.0 - y) @ valid
    s1 = y @ np.where(valid, ranks, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (s1 - n1 * (n1 + 1) / 2) / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


def shift_null(y: np.ndarray, ranks: np.ndarray, min_shift: int) -> np.ndarray:
    """The best separation |AUC - 0.5| in the pool for every allowed circular shift of y.

    Shifts closer than min_shift to zero (either way) are excluded, so with
    min_shift the longest marked run no run overlaps itself. Exact for every
    remaining shift: the shifted rank sums and marked counts are circular
    cross-correlations, computed by FFT.
    """
    n = y.size
    shifts = np.arange(min_shift, n - min_shift + 1)
    best = np.zeros(shifts.size)
    if shifts.size == 0 or ranks.shape[1] == 0:
        return best
    fy = np.conj(np.fft.rfft(y))[:, None]
    for c0 in range(0, ranks.shape[1], _FFT_CHUNK):
        block = ranks[:, c0:c0 + _FFT_CHUNK]
        valid = ~np.isnan(block)
        v = valid.astype(np.float64)
        s1 = np.fft.irfft(fy * np.fft.rfft(np.where(valid, block, 0.0), axis=0), n=n, axis=0)[shifts]
        n1 = np.rint(np.fft.irfft(fy * np.fft.rfft(v, axis=0), n=n, axis=0)[shifts])
        n0 = v.sum(axis=0) - n1
        with np.errstate(invalid="ignore", divide="ignore"):
            sep = np.abs((s1 - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5)
        sep[~np.isfinite(sep)] = 0.0
        best = np.maximum(best, sep.max(axis=1))
    return best


def exceedances(sep: float, null: np.ndarray) -> int:
    """Shifted labelings whose best column separates at least as well as sep."""
    return int((null >= sep - _TIE).sum())


def critical_separation(null: np.ndarray) -> float | None:
    """The separation a column must exceed to be beyond chance; None when the shifts are too few.

    Beyond chance needs (1 + exceedances) / (1 + shifts) <= 1/20, so at most
    k = (1 + shifts) // 20 - 1 shifts may separate as well: the column must beat
    the (k + 1)-th best shifted separation. Below 19 shifts nothing can.
    """
    k = (1 + null.size) // _BEYOND_ONE_IN - 1
    return float(np.sort(null)[::-1][k]) if k >= 0 else None


def weighted_auc(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AUC under row weights (one row of weights per draw), ties counted as half."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    w = weights[:, order]
    pos = np.add.reduceat(w * ys, starts, axis=1)
    neg = np.add.reduceat(w * (1.0 - ys), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (below + 0.5 * neg)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))


def bootstrap_interval(x: np.ndarray, y: np.ndarray, block: int, seed: int) -> tuple[float, float] | None:
    """A 90% moving-block bootstrap interval for the AUC, circular blocks of block rows.

    Draws without both groups have no AUC and are dropped; None when every draw is.
    """
    n = x.size
    rng = np.random.default_rng(seed)
    count = -(-n // block)
    starts = rng.integers(0, n, size=(_BOOTSTRAP_DRAWS, count))
    rows = ((starts[:, :, None] + np.arange(block)) % n).reshape(_BOOTSTRAP_DRAWS, -1)[:, :n]
    flat = rows + (np.arange(_BOOTSTRAP_DRAWS) * n)[:, None]
    weights = np.bincount(flat.ravel(), minlength=_BOOTSTRAP_DRAWS * n).reshape(_BOOTSTRAP_DRAWS, n)
    draws = weighted_auc(x, y, weights.astype(np.float64))
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return None
    lo, hi = np.quantile(draws, [0.05, 0.95])
    return float(lo), float(hi)


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two rank columns over the rows both define."""
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.sum() < 3:
        return 0.0
    rho = np.corrcoef(a[both], b[both])[0, 1]
    return float(rho) if np.isfinite(rho) else 0.0


def threshold(x: np.ndarray, y: np.ndarray, operator: str) -> float | None:
    """The cut that maximizes TPR - FPR for `x operator cut`; None when no cut beats zero.

    The cut keeps exactly the split it was chosen for: the midpoint between the
    last flagged and first unflagged value, at the fewest significant digits that
    still fall between them.
    """
    valid = np.isfinite(x)
    x, y = x[valid], y[valid].astype(bool)
    if not y.any() or y.all():
        return None
    order = np.argsort(-x if operator == ">" else x, kind="stable")
    xs, ys = x[order], y[order]
    youden = np.cumsum(ys) / ys.sum() - np.cumsum(~ys) / (~ys).sum()
    youden[np.r_[xs[1:] == xs[:-1], False]] = -np.inf
    k = int(np.argmax(youden))
    # Flagging every row scores exactly zero, so a positive best is never the last value.
    if youden[k] <= 0:
        return None
    return _cut(float(xs[k]), float(xs[k + 1]), operator)


def _cut(flagged: float, unflagged: float, operator: str) -> float:
    """The shortest rounding of the midpoint with `flagged operator cut` and not `unflagged operator cut`."""
    mid = flagged + (unflagged - flagged) / 2
    for digits in range(1, 18):
        cut = float(f"{mid:.{digits}g}")
        if (flagged > cut >= unflagged) if operator == ">" else (flagged < cut <= unflagged):
            return cut
    return unflagged


def rule_hits(values: np.ndarray, column: str, operator: str, value: float, persistence_n: int) -> np.ndarray:
    """Where the Limen threshold condition holds, evaluated by Limen's own predicate."""
    condition = {"id": "entry", "type": "threshold", "column": column, "operator": operator,
                 "value": value, "persistence_n": persistence_n}
    frame = pl.DataFrame({column: values}).fill_nan(None)
    return frame.select(build_predicate(condition, {}).fill_null(False)).to_series().to_numpy()


def fit_rule(
    values: np.ndarray, column: str, direction: str, marked: np.ndarray, rows: np.ndarray,
    events: list[tuple[int, int]],
) -> dict[str, Any] | None:
    """A Limen threshold rule in the column's direction, with the best persistence by TPR - FPR."""
    operator = ">" if direction == "higher" else "<"
    cut = threshold(values[rows], marked[rows], operator)
    if cut is None:
        return None
    pos, neg = marked & rows, ~marked & rows
    best = None
    for n in _PERSISTENCE:
        hits = rule_hits(values, column, operator, cut, n)
        tpr, fpr = float(hits[pos].mean()), float(hits[neg].mean())
        if best is None or tpr - fpr > best["tpr"] - best["fpr"]:
            flagged = int((hits & rows).sum())
            best = {
                "column": column, "operator": operator, "value": cut, "persistence_n": n,
                "tpr": round(tpr, 4), "fpr": round(fpr, 4),
                "precision": round(float((hits & pos).sum()) / flagged, 4) if flagged else None,
                "events_hit": round(sum(bool(hits[a:b].any()) for a, b in events) / len(events), 4),
                "_hits": hits,
            }
    return best


def split_dates(context: tuple[int, int], data: tuple[int, int]) -> tuple[dict[str, str], bool] | None:
    """Chronological splits of the series' data whose test lies after the discovery context when 60 days remain.

    `data` is the span the bar series covers, its first bar to its latest. Returns
    the split and whether its test is out of sample. Without that much time after
    the context, it falls back to a 60/20/20 split of the data's days, each at
    least one day, whose test the context may overlap. Fewer than three days have
    no such split: None.
    """
    day = timedelta(days=1)
    first, final, last = day_of(data[0]), day_of(data[1] - 1), day_of(context[1] - 1)
    remaining = (final - last).days
    if remaining >= 60:
        val_start = last + day
        test_start = val_start + timedelta(days=remaining // 2)
        dates = (first, last, val_start, test_start - day, test_start, final)
        clean = True
    else:
        days = (final - first).days + 1
        if days < 3:
            return None
        to_val = max(1, days * 6 // 10)
        to_test = min(days - 1, max(to_val + 1, days * 8 // 10))
        val_start, test_start = first + timedelta(days=to_val), first + timedelta(days=to_test)
        dates = (first, val_start - day, val_start, test_start - day, test_start, final)
        clean = last < test_start
    keys = ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")
    return dict(zip(keys, (d.isoformat() for d in dates))), clean


def manifest(
    *, bar_source: dict[str, Any], factory: str, source: str, params: dict[str, Any], rule: dict[str, Any],
    name: str, description: str, context: tuple[int, int], data: tuple[int, int],
) -> dict[str, Any] | None:
    """The fitted rule as a Limen rule-based experiment manifest, threshold as its one parameter.

    Limen derives the data window from split_dates, so the source keeps only its
    file; every transform, feature or indicator, is a rule-based indicator. None
    when the data has too few days to split.
    """
    condition = {
        "id": "entry", "name": description, "type": "threshold", "column": rule["column"],
        "operator": rule["operator"], "value": "{threshold}", "persistence_n": rule["persistence_n"],
    }
    split = split_dates(context, data)
    if split is None:
        return None
    splits, clean = split
    body: dict[str, Any] = {"type": "rule_based", "data_source": {"method": bar_source["method"], "params": dict(bar_source["params"])},
                            "split_dates": splits}
    if source != "raw":
        body["indicators"] = [{"func": f"{source}.{factory}", "params": params}]
    body["strategy"] = {"conditions": [condition], "entry": "entry"}
    if not clean:
        description += "; its test split overlaps the discovery context, so it is not out of sample"
    body["reference_architecture"] = "limen.sfd.reference_architecture.rule_based"
    return {
        "schema_version": "1.0",
        "metadata": {"name": name, "limen_version": version("vaquum_limen"), "mode": "development",
                     "description": description},
        "sfd": {"manifest": body, "params": {"threshold": [rule["value"]]}},
        "uel": {"n_permutations": 1, "search_strategy": {"type": "grid"}, "prep_each_round": True,
                "output_format": "csv"},
    }


def series(values: np.ndarray, ms: np.ndarray, step_ms: int, start_ms: int, end_ms: int,
           max_points: int) -> dict[str, Any] | None:
    """Values over [start, end), each bucket of bars carried by its last bar (known at its end)."""
    inside = np.flatnonzero((ms >= start_ms) & (ms + step_ms <= end_ms))
    if inside.size == 0:
        return None
    width = -(-inside.size // max_points)
    picks = inside[np.minimum(np.arange(width - 1, inside.size + width - 1, width), inside.size - 1)]
    picked = values[picks]
    return {
        "t0": int(ms[inside[0]]), "step_ms": int(step_ms * width), "bucket_bars": int(width),
        "values": [None if not np.isfinite(v) else round(float(v), 8) for v in picked],
    }


def spans(hits: np.ndarray, ms: np.ndarray, step_ms: int, start_ms: int, end_ms: int,
          limit: int = 2000) -> list[list[int]]:
    """Runs of rule hits inside [start, end) as [open, close) milliseconds."""
    inside = (ms >= start_ms) & (ms + step_ms <= end_ms)
    return [[int(ms[a]), int(ms[b - 1] + step_ms)] for a, b in runs(hits & inside)[:limit]]


def discover(
    pool: Pool, *, step_ms: int, windows: Sequence[tuple[int, int]], mode: str, lead_bars: int,
    context: tuple[int, int], display: tuple[int, int], top: int, max_points: int,
    bar_source: dict[str, Any], label: str,
) -> dict[str, Any]:
    """Rank the pool against the marked bars, measure chance, and describe the leaders."""
    clock0 = time.perf_counter()
    marked, left_out = target(pool.ms, step_ms, windows, mode, lead_bars)
    rows = (pool.ms >= context[0]) & (pool.ms + step_ms <= context[1]) & ~left_out
    marked &= rows
    events = runs(marked)
    if not events:
        raise ValueError("no marked bar falls inside the context")
    if not (rows & ~marked).any():
        raise ValueError("the context has no unmarked bar to compare against")
    index = np.flatnonzero(rows)
    y = marked[index].astype(np.float64)
    ranks = column_ranks(pool.values[index])
    timing = {"rank": time.perf_counter() - clock0}

    clock1 = time.perf_counter()
    point = auc(y, ranks)
    separation = np.abs(point - 0.5)
    ranked = np.flatnonzero(np.isfinite(separation))
    ranked = ranked[np.argsort(-separation[ranked], kind="stable")]
    timing["screen"] = time.perf_counter() - clock1

    clock2 = time.perf_counter()
    longest = max(b - a for a, b in runs(y.astype(bool)))
    min_shift = longest
    null = shift_null(y, ranks, min_shift)
    timing["null"] = time.perf_counter() - clock2
    critical = critical_separation(null)

    clock3 = time.perf_counter()
    intervals = len(events) >= MIN_EVENTS_FOR_INTERVALS
    leaders: list[dict[str, Any]] = []
    kept: list[int] = []
    for j in ranked[:_SCREEN_LEADERS]:
        twin = next((k for k in kept if abs(rank_correlation(ranks[:, j], ranks[:, k])) > _CLUSTER_RHO), None)
        if twin is not None:
            leaders[kept.index(twin)]["similar"].append(pool.names[j])
            continue
        if len(kept) == MAX_LEADERS:
            continue
        kept.append(int(j))
        leaders.append(_leader(pool, int(j), point[j], separation[j], null, y, ranks, intervals))
    # Leaders with an interval rank by its conservative end, the rest after them by point separation.
    # top takes a prefix of that order over the same clusters, so asking for fewer never changes who leads.
    order = sorted(range(len(leaders)), key=lambda i: (
        leaders[i]["conservative"] is None, -(leaders[i]["conservative"] or 0.0), -leaders[i]["separation"]))
    leaders = [_describe(leaders[i], pool, kept[i], marked, rows, events, step_ms, context, display, max_points,
                         bar_source, label) for i in order[:top]]
    timing["stage_two"] = time.perf_counter() - clock3
    timing["total"] = time.perf_counter() - clock0

    return {
        "target": {"mode": mode, "lead_bars": lead_bars if mode == "lead" else None,
                   "windows": [list(w) for w in windows], "marked_bars": int(marked.sum()),
                   "events": len(events), "longest_run": int(longest)},
        "context": {"start_ms": int(context[0]), "end_ms": int(context[1]), "bars": int(index.size),
                    "unmarked_bars": int(index.size - marked.sum())},
        "evidence": "events" if len(events) > 1 else "description",
        "intervals": intervals,
        "chance": {"method": "circular-shift", "shifts": int(null.size), "min_shift": int(min_shift),
                   "alpha": 1 / _BEYOND_ONE_IN, "critical": None if critical is None else round(critical, 4),
                   "best": round(float(separation[ranked[0]]), 4) if ranked.size else None},
        "screened": int(ranked.size),
        "timing_ms": {k: round(v * 1000, 2) for k, v in timing.items()},
        "candidates": leaders,
    }


def _leader(
    pool: Pool, j: int, point: float, sep: float, null: np.ndarray, y: np.ndarray, ranks: np.ndarray,
    intervals: bool,
) -> dict[str, Any]:
    """A cluster leader's separation, chance call and interval: everything its rank depends on."""
    column = pool.names[j]
    direction = "higher" if point > 0.5 else "lower"
    over = exceedances(sep, null)
    out: dict[str, Any] = {
        "column": column, "factory": pool.factories[j], "source": pool.sources[j], "params": pool.params[j],
        "auc": round(float(point), 4), "direction": direction, "separation": round(float(sep), 4),
        "p_adjusted": round((1 + over) / (1 + null.size), 4),
        "beyond_chance": _BEYOND_ONE_IN * (1 + over) <= 1 + null.size,
        "ci90": None, "conservative": None, "similar": [],
    }
    valid = ~np.isnan(ranks[:, j])
    # The column's own events: marked runs among the rows it defines, as its bootstrap sees them.
    own = runs(y[valid].astype(bool)) if intervals else []
    if len(own) >= MIN_EVENTS_FOR_INTERVALS:
        block = max(max(b - a for a, b in own), _MIN_BLOCK)
        interval = bootstrap_interval(ranks[valid, j], y[valid], block, seed=j)
        if interval is not None:
            lo, hi = interval
            out["ci90"] = [round(lo, 4), round(hi, 4)]
            out["conservative"] = round(max(0.0, lo - 0.5) if direction == "higher" else max(0.0, 0.5 - hi), 4)
    return out


def _describe(
    out: dict[str, Any], pool: Pool, j: int, marked: np.ndarray, rows: np.ndarray,
    events: list[tuple[int, int]], step_ms: int, context: tuple[int, int], display: tuple[int, int],
    max_points: int, bar_source: dict[str, Any], label: str,
) -> dict[str, Any]:
    """A returned leader's fitted rule, its hits and manifest, and its series."""
    column, values = pool.names[j], pool.values[:, j]
    rule = fit_rule(values, pool.natives[j], out["direction"], marked, rows, events)
    if rule is not None:
        hits = rule.pop("_hits")
        value = np.format_float_positional(rule["value"], trim="-")
        text = f"{column} {rule['operator']} {value} for {rule['persistence_n']} bar" + ("s" if rule["persistence_n"] > 1 else "")
        rule["hits"] = spans(hits, pool.ms, step_ms, *display)
        rule["manifest"] = manifest(bar_source=bar_source, factory=pool.factories[j], source=pool.sources[j],
                                    params=pool.params[j], rule=rule, name=f"find_features_{label}",
                                    description=f"{text}, fitted in sample by Mill Find features", context=context,
                                    data=(int(pool.ms[0]), int(pool.ms[-1]) + step_ms))
        rule["text"] = text
    out["rule"] = rule
    out["series"] = series(values, pool.ms, step_ms, *display, max_points)
    return out
