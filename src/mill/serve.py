from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import polars as pl

from . import api_reference, feature_registry, grading, qc, targets
from .bars import DOLLAR_BAR_USD, TIME_BAR_SECONDS, arrow_series
from .settings import (
    APP_NAME,
    ARROW_LATEST_NAME,
    DEFAULT_ARROW_DIR,
    DEFAULT_BAR,
    PROJECT_END_STR,
    PROJECT_START_STR,
    config_root,
    host,
    port,
)
from .table import build_from_spec, klines
from .views import ALL as VIEWS

_INDEX = files("mill").joinpath("assets", "index.html")
_TAGS_PATH = config_root() / "feature_tags.json"
_SAVED_VIEWS_PATH = config_root() / "views.json"
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ROW_MODES = {
    "bar", "all", "day", "week", "month", "quarter", "year",
    "dow", "dom", "woy", "cut", "quantile", "decile",
}
_STAT_ROW_MODES = {"cut", "quantile", "decile"}
_QUANTILE_PCTS = (0.01, 5, 25, 50, 75, 95, 99.9)

_BARS = tuple(TIME_BAR_SECONDS) + tuple(DOLLAR_BAR_USD)
_TARGETS = ("none", *targets.NAMES)
_ARROW_DATA_SOURCE_METHOD = "limen.data.HistoricalData.get_arrow_file"


def _arrow_data_source(bar: str) -> dict[str, object]:
    # Canonical deploy path: the read-only tdw Arrow volume is mounted at
    # /opt/arrow in both Mill and the Limen runner that consumes the manifest,
    # independent of any local MILL_ARROW_DIR override used for tests.
    file_path = f"{DEFAULT_ARROW_DIR}/{arrow_series(bar)}/{ARROW_LATEST_NAME}"
    return {
        "method": _ARROW_DATA_SOURCE_METHOD,
        "params": {
            "file_path": file_path,
            "start_date_limit": PROJECT_START_STR,
            "end_date_limit": PROJECT_END_STR,
        },
    }


_BAR_DATA_SOURCES: dict[str, dict[str, object]] = {
    name: _arrow_data_source(name) for name in _BARS
}
_TARGET_CLASSES: dict[str, dict[str, object]] = {
    name: targets.binding(name) for name in targets.NAMES
}
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")

# Static description of the endpoint surface, served at GET /api-reference. The
# enum values are taken from the live route constants so the document tracks
# what the endpoints actually accept; serialized once since it never changes
# after startup.
_API_REFERENCE = api_reference.build_reference(
    app_name=APP_NAME,
    bars=list(_BARS),
    bar_default=_BARS[0],
    targets=list(_TARGETS),
    row_modes=sorted(_ROW_MODES),
    data_window={"start": PROJECT_START_STR, "end": PROJECT_END_STR},
)
_API_REFERENCE_BYTES = json.dumps(_API_REFERENCE).encode()


_PAYLOADS: dict[tuple[str, str, str], bytes] = {}
_FRAMES: dict[tuple[str, str, str], pl.DataFrame] = {}
_QC_CACHE: dict[tuple[str, str, str, tuple[str, ...], str, str, str, str], bytes] = {}
_RAW_CACHE: dict[str, pl.DataFrame] = {}
_BAR_LOCKS: dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()

_VIEW_SPECS: dict[str, list[dict[str, Any]]] = {}
_VIEW_TARGETS: dict[str, str] = {}
_VIEW_BARS: dict[str, list[str]] = {}
_FACTORY_TAGS: dict[str, list[str]] = {}
_BUILTIN_VIEWS: tuple[str, ...] = ()
_BUILTIN_VIEW_NAMES: set[str] = set()
_SAVED_VIEWS: dict[str, dict[str, Any]] = {}
_APP_USERNAME = ""
_APP_PASSWORD = ""
_API_SECRET = ""


def _require_auth_env() -> None:
    global _APP_USERNAME, _APP_PASSWORD, _API_SECRET
    _APP_USERNAME = os.environ.get("APP_USERNAME", "")
    _APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
    _API_SECRET = os.environ.get("API_SECRET", "")
    if not (_APP_USERNAME and _APP_PASSWORD and _API_SECRET):
        raise RuntimeError(
            "Mill requires APP_USERNAME, APP_PASSWORD, and "
            "API_SECRET environment variables; refusing to start"
        )


def _load_factory_tags() -> dict[str, list[str]]:
    if not _TAGS_PATH.exists():
        return {}
    try:
        data = json.loads(_TAGS_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        k: [t for t in v if isinstance(t, str)]
        for k, v in data.items()
        if isinstance(v, list)
    }


def _save_factory_tags() -> None:
    _TAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TAGS_PATH.write_text(
        json.dumps(_FACTORY_TAGS, indent=2, sort_keys=True) + "\n"
    )


def _load_saved_views() -> dict[str, dict[str, Any]]:
    if not _SAVED_VIEWS_PATH.exists():
        return {}
    try:
        data = json.loads(_SAVED_VIEWS_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, item in data.items():
        if not _NAME_RE.fullmatch(name) or not isinstance(item, dict):
            continue
        spec = item.get("spec")
        target = item.get("target", "none")
        bars = item.get("bars", list(_BARS))
        if (
            isinstance(spec, list)
            and _valid_spec(spec)
            and target in _TARGETS
            and isinstance(bars, list)
            and all(isinstance(b, str) and b in _BARS for b in bars)
        ):
            out[name] = {"spec": spec, "target": target, "bars": bars}
    return out


def _save_saved_views() -> None:
    _SAVED_VIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SAVED_VIEWS_PATH.write_text(
        json.dumps(_SAVED_VIEWS, indent=2, sort_keys=True) + "\n"
    )


def _sanitize_tags(raw: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        t = item.strip().lstrip("#").lower()
        if t and t not in seen and _TAG_RE.match(t):
            seen.add(t)
            out.append(t)
    return out


def _spec_hash(spec: list[dict[str, Any]]) -> str:
    canon = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canon.encode()).hexdigest()[:16]


def _finite_round(name: str) -> pl.Expr:
    return (
        pl.when(pl.col(name).is_finite())
        .then(pl.col(name).round(4))
        .otherwise(None)
        .alias(name)
    )


def _serialize(df: pl.DataFrame) -> bytes:
    feat_cols = [c for c in df.columns if c != "datetime"]
    numeric = {c for c, dt in df.schema.items() if dt.is_float()}
    feat_exprs = [
        _finite_round(c) if c in numeric else pl.col(c)
        for c in feat_cols
    ]
    out = df.select(
        pl.col("datetime").dt.to_string("%Y-%m-%d %H:%M").alias("datetime"),
        *feat_exprs,
    )
    return json.dumps(
        {"columns": out.columns, "rows": out.rows()},
        allow_nan=False,
    ).encode()


def _raw_for(bar: str) -> pl.DataFrame:
    raw = _RAW_CACHE.get(bar)
    if raw is None:
        raw = klines(bar)
        _RAW_CACHE[bar] = raw
    return raw


def _date_filter(df: pl.DataFrame, date_from: str, date_to: str) -> pl.DataFrame:
    if "datetime" not in df.columns or not (date_from or date_to):
        return df
    day = pl.col("datetime").dt.date()
    if date_from:
        df = df.filter(day >= pl.lit(date_from).str.to_date())
    if date_to:
        df = df.filter(day <= pl.lit(date_to).str.to_date())
    return df


def _median_exprs(df: pl.DataFrame) -> list[pl.Expr]:
    return [
        pl.col(c).median().alias(c)
        for c, dtype in df.schema.items()
        if c not in {"datetime", "_bucket"} and dtype.is_numeric()
    ]


def _group_for_qc(
    df: pl.DataFrame,
    mode: str,
    cut_n: int,
    target: str | None,
) -> pl.DataFrame:
    if mode == "bar":
        return df
    exprs = _median_exprs(df)
    if not exprs:
        return df
    if mode == "all":
        return df.select(pl.lit("all").alias("datetime"), *exprs)
    if mode in _STAT_ROW_MODES:
        return _stat_group_for_qc(df, mode, cut_n, target)
    bucket = _time_bucket(mode)
    if bucket is None:
        return df
    return (
        df.with_columns(bucket.alias("_bucket"))
        .group_by("_bucket")
        .agg(exprs)
        .sort("_bucket")
        .rename({"_bucket": "datetime"})
    )


def _time_bucket(mode: str) -> pl.Expr | None:
    dt = pl.col("datetime")
    if mode == "day":
        return dt.dt.strftime("%Y-%m-%d")
    if mode == "week":
        return dt.dt.strftime("%G-W%V")
    if mode == "month":
        return dt.dt.strftime("%Y-%m")
    if mode == "quarter":
        return (
            dt.dt.year().cast(pl.Utf8)
            + "-Q"
            + dt.dt.quarter().cast(pl.Utf8)
        )
    if mode == "year":
        return dt.dt.year().cast(pl.Utf8)
    if mode == "dow":
        return dt.dt.weekday().cast(pl.Utf8)
    if mode == "dom":
        return dt.dt.day().cast(pl.Utf8).str.zfill(2)
    if mode == "woy":
        return dt.dt.week().cast(pl.Utf8).str.zfill(2)
    return None


def _stat_group_for_qc(
    df: pl.DataFrame,
    mode: str,
    cut_n: int,
    target: str | None,
) -> pl.DataFrame:
    anchor = target if target and target in df.columns else _first_value_col(df)
    if anchor is None:
        return df
    values = df[anchor].cast(pl.Float64, strict=False).to_numpy()
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return df.clear()
    sorted_values = np.sort(finite)
    cuts, labels = _stat_cuts(mode, sorted_values, cut_n)
    if not labels:
        return df.clear()
    bucket, order = _bucket_labels(values, cuts, labels)
    exprs = _median_exprs(df)
    return (
        df.with_columns(
            pl.Series("_bucket", bucket),
            pl.Series("_bucket_order", order),
        )
        .filter(pl.col("_bucket").is_not_null())
        .group_by("_bucket_order", "_bucket")
        .agg(exprs)
        .sort("_bucket_order")
        .drop("_bucket_order")
        .rename({"_bucket": "datetime"})
    )


def _first_value_col(df: pl.DataFrame) -> str | None:
    for c, dtype in df.schema.items():
        if c != "datetime" and dtype.is_numeric():
            return c
    return None


def _stat_cuts(
    mode: str,
    sorted_values: np.ndarray,
    cut_n: int,
) -> tuple[list[float], list[str]]:
    if mode == "decile":
        cuts = [
            float(sorted_values[int(k * sorted_values.size / 10)])
            for k in range(1, 10)
        ]
        labels = [f"d{i:02d}" for i in range(1, 11)]
        return cuts, labels
    if mode == "quantile":
        centers = [
            float(sorted_values[min(int(p * sorted_values.size / 100), sorted_values.size - 1)])
            for p in _QUANTILE_PCTS
        ]
        cuts = [
            (centers[i] + centers[i + 1]) / 2
            for i in range(len(centers) - 1)
        ]
        labels = [f"p{p:g}" for p in _QUANTILE_PCTS]
        return cuts, labels
    n = max(2, int(cut_n))
    low = float(sorted_values[0])
    high = float(sorted_values[-1])
    if low == high:
        return [], []
    width = (high - low) / n
    cuts = [low + k * width for k in range(1, n)]
    labels = [
        f"[{low + i * width:.2f},{low + (i + 1) * width:.2f})"
        for i in range(n)
    ]
    return cuts, labels


def _bucket_labels(
    values: np.ndarray,
    cuts: list[float],
    labels: list[str],
) -> tuple[list[str | None], list[int | None]]:
    bucket: list[str | None] = []
    order: list[int | None] = []
    for value in values:
        if not np.isfinite(value):
            bucket.append(None)
            order.append(None)
            continue
        idx = int(np.searchsorted(cuts, value, side="right"))
        bucket.append(labels[idx])
        order.append(idx)
    return bucket, order


def _ensure(
    bar: str, target: str, spec: list[dict[str, Any]]
) -> tuple[bytes, pl.DataFrame]:
    sh = _spec_hash(spec)
    key = (bar, target, sh)
    payload = _PAYLOADS.get(key)
    if payload is not None:
        return payload, _FRAMES[key]
    with _REGISTRY_LOCK:
        bar_lock = _BAR_LOCKS.setdefault(bar, threading.Lock())
    with bar_lock:
        payload = _PAYLOADS.get(key)
        if payload is not None:
            return payload, _FRAMES[key]
        raw = _raw_for(bar)
        if spec:
            features = build_from_spec(raw, spec)
        else:
            features = raw.select(pl.col("datetime"))
        if target == "none":
            result = features
        else:
            tdf = targets.apply(raw, target).select(
                pl.col("datetime"), pl.col(target)
            )
            result = features.join(tdf, on="datetime", how="left")
        payload = _serialize(result)
        _PAYLOADS[key] = payload
        _FRAMES[key] = result
    return payload, result


def _parse_features(q: dict[str, list[str]]) -> tuple[list[dict[str, Any]], str | None]:
    raw = q.get("features", ["[]"])[0]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], f"bad features json: {exc}"
    if not isinstance(parsed, list):
        return [], "features must be a list"
    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            return [], "each feature must be an object"
        factory = item.get("factory")
        if factory not in feature_registry.REGISTRY:
            return [], f"unknown factory: {factory!r}"
        params = item.get("params") or {}
        if not isinstance(params, dict):
            return [], "params must be an object"
        err = _validate_params(factory, params)
        if err is not None:
            return [], err
        out.append({"factory": factory, "params": params})
    return out, None


def _valid_param_value(value: object, type_: str) -> bool:
    if type_ == "bool":
        return isinstance(value, bool)
    if type_ == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_ == "float":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
        )
    if type_ == "str":
        return isinstance(value, str) and bool(value)
    return False


def _validate_params(factory: str, params: dict[str, object]) -> str | None:
    specs = {
        spec["name"]: spec["type"]
        for spec in feature_registry.params_for(factory)
    }
    for name, value in params.items():
        if name not in specs:
            return f"unknown param for {factory}: {name!r}"
        if not _valid_param_value(value, specs[name]):
            return f"invalid {specs[name]} param for {factory}: {name!r}"
    return None


def _valid_spec(spec: list[Any]) -> bool:
    for item in spec:
        if not isinstance(item, dict):
            return False
        factory = item.get("factory")
        params = item.get("params") or {}
        if factory not in feature_registry.REGISTRY or not isinstance(params, dict):
            return False
        if _validate_params(factory, params) is not None:
            return False
    return True


def _parse_selectors(
    q: dict[str, list[str]],
) -> tuple[str, str, bytes | None]:
    bar = q.get("bars", [_BARS[0]])[0]
    target = q.get("target", ["none"])[0]
    if bar not in _BARS or target not in _TARGETS:
        return bar, target, b"bad bars/target"
    return bar, target, None


def _save_view(
    name: str, spec: list[dict[str, Any]], target: str
) -> tuple[int, str]:
    if not _NAME_RE.fullmatch(name):
        return 400, (
            "name must be lowercase letters, digits, underscores; "
            "start with a letter; max 41 chars"
        )
    if target not in _TARGETS:
        return 400, f"unknown target: {target!r}"
    if name in _BUILTIN_VIEW_NAMES:
        return 409, f"cannot overwrite built-in view: {name!r}"
    with _REGISTRY_LOCK:
        _SAVED_VIEWS[name] = {"spec": spec, "target": target, "bars": list(_BARS)}
        _save_saved_views()
        _VIEW_SPECS[name] = spec
        _VIEW_TARGETS[name] = target
        _VIEW_BARS[name] = list(_BARS)
    return 200, "ok"


def _delete_view(name: str) -> tuple[int, str]:
    if not _NAME_RE.fullmatch(name):
        return 400, "invalid view name"
    with _REGISTRY_LOCK:
        if name in _BUILTIN_VIEW_NAMES:
            return 409, f"cannot delete built-in view: {name!r}"
        if name not in _SAVED_VIEWS:
            return 404, f"unknown view: {name!r}"
        _SAVED_VIEWS.pop(name, None)
        _save_saved_views()
        _VIEW_SPECS.pop(name, None)
        _VIEW_TARGETS.pop(name, None)
        _VIEW_BARS.pop(name, None)
    return 200, "ok"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if not self._authorized(url.path):
            self._unauthorized()
            return
        if url.path == "/health":
            self._send(200, "application/json", b'{"status":"ok"}')
        elif url.path == "/":
            self._send(200, "text/html; charset=utf-8", _INDEX.read_bytes())
        elif url.path == "/app-config":
            with _REGISTRY_LOCK:
                body = json.dumps(
                    {
                        "bars": list(_BARS),
                        "default_bar": DEFAULT_BAR,
                        "app_name": APP_NAME,
                        "data_window": {
                            "start": PROJECT_START_STR,
                            "end": PROJECT_END_STR,
                        },
                        "views": list(_VIEW_SPECS.keys()),
                        "saved_views": list(_SAVED_VIEWS.keys()),
                        "builtin_views": list(_BUILTIN_VIEWS),
                        "view_specs": _VIEW_SPECS,
                        "view_targets": _VIEW_TARGETS,
                        "view_bars": _VIEW_BARS,
                        "targets": list(_TARGETS),
                        "factories": feature_registry.catalog(),
                        "factory_tags": _FACTORY_TAGS,
                        "bar_data_sources": _BAR_DATA_SOURCES,
                        "target_classes": _TARGET_CLASSES,
                    }
                ).encode()
            self._send(200, "application/json", body)
        elif url.path == "/version":
            body = json.dumps(
                {
                    "sha": os.environ.get("GIT_SHA", ""),
                    "date": os.environ.get("BUILD_DATE", ""),
                    "subject": os.environ.get("GIT_SUBJECT", ""),
                }
            ).encode()
            self._send(200, "application/json", body)
        elif url.path == "/api-reference":
            self._send(200, "application/json", _API_REFERENCE_BYTES)
        elif url.path == "/data":
            q = parse_qs(url.query)
            bar, target, err = _parse_selectors(q)
            if err is not None:
                self._send(400, "text/plain", err)
                return
            spec, ferr = _parse_features(q)
            if ferr is not None:
                self._send(400, "text/plain", ferr.encode())
                return
            try:
                payload, _frame = _ensure(bar, target, spec)
            except Exception as exc:
                self._send(500, "text/plain", f"build failed: {exc}".encode())
                return
            self._send(200, "application/json", payload)
        elif url.path == "/qc":
            q = parse_qs(url.query)
            bar, target, err = _parse_selectors(q)
            if err is not None:
                self._send(400, "text/plain", err)
                return
            spec, ferr = _parse_features(q)
            if ferr is not None:
                self._send(400, "text/plain", ferr.encode())
                return
            cols_str = q.get("cols", [""])[0]
            cols = tuple(c for c in cols_str.split(",") if c)
            date_from = q.get("from", [""])[0]
            date_to = q.get("to", [""])[0]
            row_mode = q.get("row_mode", ["bar"])[0]
            if row_mode not in _ROW_MODES:
                self._send(400, "text/plain", b"bad row_mode")
                return
            try:
                cut_n = max(2, int(q.get("cut_n", ["5"])[0]))
            except ValueError:
                self._send(400, "text/plain", b"bad cut_n")
                return
            sh = _spec_hash(spec)
            cache_key = (
                bar, target, sh, cols, date_from, date_to, row_mode, str(cut_n)
            )
            cached = _QC_CACHE.get(cache_key)
            if cached is not None:
                self._send(200, "application/json", cached)
                return
            try:
                _payload, frame = _ensure(bar, target, spec)
                if cols:
                    sel = ["datetime", *(c for c in cols if c in frame.columns)]
                    if target != "none" and target in frame.columns:
                        sel.append(target)
                    sub = frame.select(sel)
                else:
                    sub = frame
                sub = _date_filter(sub, date_from, date_to)
                sub = _group_for_qc(
                    sub,
                    row_mode,
                    cut_n,
                    target if target != "none" else None,
                )
                report = qc.report(sub, target if target != "none" else None)
                report["grades"] = grading.grade_map(
                    report, target=target if target != "none" else None
                )
            except Exception as exc:
                self._send(500, "text/plain", f"qc failed: {exc}".encode())
                return
            try:
                body = json.dumps(report, allow_nan=False).encode()
            except ValueError as exc:
                self._send(500, "text/plain", f"qc serialize failed: {exc}".encode())
                return
            _QC_CACHE[cache_key] = body
            self._send(200, "application/json", body)
        else:
            self._send(404, "text/plain", b"")

    def do_POST(self) -> None:
        url = urlsplit(self.path)
        if not self._authorized(url.path):
            self._unauthorized()
            return
        if url.path == "/save_view":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode() or "{}")
            except json.JSONDecodeError as exc:
                self._send(400, "text/plain", f"bad json: {exc}".encode())
                return
            name = body.get("name", "")
            spec = body.get("spec", [])
            target = body.get("target", "none")
            if not isinstance(spec, list):
                self._send(400, "text/plain", b"spec must be a list")
                return
            if not _valid_spec(spec):
                self._send(400, "text/plain", b"invalid spec")
                return
            status, msg = _save_view(name, spec, target)
            body = {"ok": status == 200, "msg": msg, "name": name}
            self._send(status, "application/json", json.dumps(body).encode())
        elif url.path == "/delete_view":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode() or "{}")
            except json.JSONDecodeError as exc:
                self._send(400, "text/plain", f"bad json: {exc}".encode())
                return
            name = body.get("name", "")
            status, msg = _delete_view(name)
            body = {"ok": status == 200, "msg": msg, "name": name}
            self._send(status, "application/json", json.dumps(body).encode())
        elif url.path == "/save_tags":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode() or "{}")
            except json.JSONDecodeError as exc:
                self._send(400, "text/plain", f"bad json: {exc}".encode())
                return
            factory = body.get("factory")
            raw_tags = body.get("tags", [])
            if factory not in feature_registry.REGISTRY:
                self._send(400, "text/plain", f"unknown factory: {factory!r}".encode())
                return
            if not isinstance(raw_tags, list):
                self._send(400, "text/plain", b"tags must be a list")
                return
            cleaned = _sanitize_tags(raw_tags)
            with _REGISTRY_LOCK:
                if cleaned:
                    _FACTORY_TAGS[factory] = cleaned
                else:
                    _FACTORY_TAGS.pop(factory, None)
                _save_factory_tags()
            self._send(200, "application/json", json.dumps({"ok": True, "tags": cleaned}).encode())
        else:
            self._send(404, "text/plain", b"")

    def do_HEAD(self) -> None:
        url = urlsplit(self.path)
        if not self._authorized(url.path):
            self._unauthorized(head=True)
            return
        if url.path in {"/", "/health"}:
            self._send(200, "text/plain", b"", head=True)
        else:
            self._send(404, "text/plain", b"", head=True)

    def log_message(self, *args: object, **kwargs: object) -> None:
        return

    def _authorized(self, path: str) -> bool:
        if path in ("/health", "/api-reference"):
            return True
        if self.command == "HEAD" and path == "/":
            return True
        auth = self.headers.get("authorization", "")
        scheme, _, credentials = auth.partition(" ")
        scheme_lc = scheme.lower()
        if scheme_lc == "basic":
            try:
                decoded = base64.b64decode(credentials, validate=True).decode(
                    "utf-8", errors="strict"
                )
            except (binascii.Error, ValueError, UnicodeDecodeError):
                decoded = ""
            user, sep, password = decoded.partition(":")
            return (
                bool(sep)
                and secrets.compare_digest(user, _APP_USERNAME)
                and secrets.compare_digest(password, _APP_PASSWORD)
            )
        if scheme_lc == "bearer":
            return secrets.compare_digest(credentials, _API_SECRET)
        return False

    def _unauthorized(self, *, head: bool = False) -> None:
        body = b'{"detail":"authentication required"}'
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("WWW-Authenticate", 'Basic realm="Mill"')
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _send(
        self,
        status: int,
        ctype: str,
        body: bytes,
        *,
        head: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)


def main() -> None:
    global _BUILTIN_VIEWS, _BUILTIN_VIEW_NAMES, _FACTORY_TAGS, _SAVED_VIEWS
    global _VIEW_SPECS, _VIEW_TARGETS, _VIEW_BARS
    _require_auth_env()
    _BUILTIN_VIEWS = tuple(v.NAME for v in VIEWS)
    _BUILTIN_VIEW_NAMES = set(_BUILTIN_VIEWS)
    _VIEW_SPECS = {v.NAME: list(v.SPEC) for v in VIEWS}
    _VIEW_TARGETS = {v.NAME: getattr(v, "TARGET", "none") for v in VIEWS}
    _VIEW_BARS = {v.NAME: list(getattr(v, "BARS", _BARS)) for v in VIEWS}
    _SAVED_VIEWS = {
        name: item
        for name, item in _load_saved_views().items()
        if name not in _BUILTIN_VIEW_NAMES
    }
    for name, item in _SAVED_VIEWS.items():
        _VIEW_SPECS[name] = item["spec"]
        _VIEW_TARGETS[name] = item["target"]
        _VIEW_BARS[name] = item["bars"]
    _FACTORY_TAGS = _load_factory_tags()
    bind_host = host()
    bind_port = port()
    url = f"http://{bind_host}:{bind_port}"
    print(
        f"{len(_BARS)} bars x {len(_VIEW_SPECS)} views x {len(_TARGETS)} targets"
        f"  ->  {url}",
        flush=True,
    )
    server = ThreadingHTTPServer((bind_host, bind_port), _Handler)
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
