from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from .frames import read_parquet, write_parquet
from .settings import cache_root


def cache_path(label: str, criteria: dict[str, object]) -> Path:
    digest = hashlib.sha256(
        json.dumps(criteria, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return cache_root() / f"{label}_{digest}.parquet"


def read_fresh(path: Path, source: Path) -> pl.DataFrame | None:
    """The cached frame while it is current: written since its source last changed.

    The bar store rewrites a series' file as bars arrive, so a cache older than
    its file is stale. Without the file to compare, the cache is the last known.
    """
    if not path.exists():
        return None
    try:
        if path.stat().st_mtime < source.stat().st_mtime:
            return None
    except FileNotFoundError:
        pass
    try:
        return read_parquet(path)
    except Exception:
        return None


def write_json_atomic(path: Path, obj: object) -> None:
    """Dump to a sibling tmp file, then rename onto the target.

    A crash at any instant leaves the previous valid file; readers never see
    a torn write. The dump fails before the tmp exists for unserializable
    input, so the target is untouched on error.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_cache(
    path: Path,
    data: pl.DataFrame,
    criteria: dict[str, object],
) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_parquet(data, tmp)
    tmp.replace(path)
    metadata = {
        "wrote_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criteria": criteria,
        "rows": data.height,
        "columns": data.columns,
    }
    write_json_atomic(path.with_suffix(".json"), metadata)
    return path
