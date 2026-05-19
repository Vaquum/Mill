from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from frames import read_parquet, write_parquet


TTL_SECONDS = 86400
CriteriaValue = str | int | float | None


def cache_root() -> Path:
    value = os.environ.get("THE_LAB_CACHE_DIR")
    if value is not None:
        return Path(value)
    return Path.home() / ".cache" / "the-lab"


def cache_path(label: str, criteria: dict[str, CriteriaValue]) -> Path:
    payload = json.dumps(criteria, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    filename = (
        f"{_safe(label)}_"
        f"{_safe(criteria.get('start_date_limit'))}_"
        f"{_safe(criteria.get('end_date_limit'))}_"
        f"{_limit_label(criteria)}_"
        f"{digest}.parquet"
    )
    return cache_root() / filename


def read_fresh(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime >= TTL_SECONDS:
        return None
    return read_parquet(path)


def write_cache(path: Path, data: pl.DataFrame, criteria: dict[str, CriteriaValue]) -> Path:
    out = write_parquet(data, path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parquet": out.name,
        "criteria": criteria,
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out


def _limit_label(criteria: dict[str, CriteriaValue]) -> str:
    for key in ("row_count_limit", "n_rows"):
        value = criteria.get(key)
        if value is not None:
            return f"{key}_{value}"
    return "all"


def _safe(value: object) -> str:
    if value is None:
        return "all"
    return str(value).replace(":", "").replace(" ", "_").replace("/", "-")
