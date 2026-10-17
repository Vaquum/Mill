from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .frames import read_parquet
from .settings import cache_root


def cache_path(label: str, criteria: dict[str, object]) -> Path:
    digest = hashlib.sha256(
        json.dumps(criteria, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return cache_root() / f"{label}_{digest}.parquet"


def read_fresh(path: Path, source: Path) -> pl.DataFrame | None:
    """The cached frame while it is current: its source has not changed since the version it was read from.

    The bar store rewrites a series' file as bars arrive. A cache carries the
    source's modification time as read (write_cache's stamp), so a later write
    makes it stale, one during the read included. Without the file to compare,
    the cache is the last known.
    """
    candidates = [p for p in (path.with_suffix(".arrow"), path) if p.exists()]
    candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for candidate in candidates:
        data = _read_candidate(candidate, source)
        if data is not None:
            return data
    return None


def file_identity(path: Path):
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except FileNotFoundError:
        return None


def _read_candidate(path: Path, source: Path) -> pl.DataFrame | None:
    try:
        identity = file_identity(path)
        current = file_identity(source)
        if current is not None:
            metadata = json.loads(path.with_suffix(".json").read_text())
            if (metadata.get("source_identity") != list(current)
                    or metadata.get("cache_identity") != list(identity)):
                return None
        data = pl.read_ipc(path, memory_map=True) if path.suffix == ".arrow" else read_parquet(path)
        if file_identity(path) != identity:
            return None
        return data
    except Exception:
        return None


@contextmanager
def _atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".mill-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, obj: object) -> None:
    """Dump to a sibling tmp file, then rename onto the target.

    A crash at any instant leaves the previous valid file; readers never see
    a torn write. The dump fails before the tmp exists for unserializable
    input, so the target is untouched on error.
    """
    payload = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    with _atomic_path(path) as temporary:
        temporary.write_text(payload)


def write_cache(
    path: Path,
    data: pl.DataFrame,
    criteria: dict[str, object],
    stamp_ns: int | None = None,
    source_identity: tuple[int, int, int] | None = None,
) -> Path:
    """Write the frame atomically; with stamp_ns, the file carries that modification time (its source's as read)."""
    path = path.with_suffix(".arrow")
    with _atomic_path(path) as temporary:
        data.write_ipc(temporary, compression="uncompressed")
        if stamp_ns is not None:
            os.utime(temporary, ns=(stamp_ns, stamp_ns))
        identity = file_identity(temporary)
    metadata = {
        "wrote_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criteria": criteria,
        "rows": data.height,
        "columns": data.columns,
        "source_identity": source_identity,
        "cache_identity": identity,
    }
    write_json_atomic(path.with_suffix(".json"), metadata)
    return path
