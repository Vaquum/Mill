from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from frames import read_parquet, write_parquet

TTL_SECONDS = 86_400


def cache_root() -> Path:
    override = os.environ.get("THE_LAB_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "the-lab"


def cache_path(label: str, criteria: dict[str, object]) -> Path:
    digest = hashlib.sha256(
        json.dumps(criteria, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return cache_root() / f"{label}_{digest}.parquet"


def read_fresh(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime >= TTL_SECONDS:
        return None
    try:
        return read_parquet(path)
    except Exception:
        return None


def write_cache(
    path: Path,
    data: pl.DataFrame,
    criteria: dict[str, object],
) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_parquet(data, tmp)
    os.replace(tmp, path)
    manifest = {
        "wrote_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criteria": criteria,
        "rows": data.height,
        "columns": data.columns,
    }
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return path
