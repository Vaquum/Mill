"""Resumable catalog construction and disk-backed pools for the research worker."""
from __future__ import annotations

import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import polars as pl

from . import discover, feature_registry
from .cache_store import _atomic_path, write_json_atomic
from .settings import cache_root, memory_budget


class PoolPending(RuntimeError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _revision():
    return _hash({"limen_version": version("vaquum_limen"),
                  "limen": distribution("vaquum_limen").read_text("direct_url.json"),
                  "polars": version("polars"),
                  "wrapper": Path(feature_registry.__file__).read_text(),
                  "builder": Path(__file__).read_text()})


def _load(directory):
    metadata = json.loads((directory / "ready.json").read_text())
    fields = {k: tuple(v) for k, v in metadata.items()}
    values = np.load(directory / "values.npy", allow_pickle=False, mmap_mode="r")
    ms = np.load(directory / "ms.npy", allow_pickle=False, mmap_mode="r")
    if values.shape != (ms.size, len(fields["names"])):
        raise ValueError("invalid persisted pool shape")
    os.utime(directory, None)
    return discover.Pool(**fields, ms=ms, values=values)


def _entries(spec):
    entries = []
    given = ([(name, {}) for name, entry in feature_registry.REGISTRY.items()
              if all(p["default"] is not None for p in entry["params"])] if spec is None
             else [(item["factory"], item.get("params") or {}) for item in spec])
    for factory, parameters in given:
        entry = feature_registry.REGISTRY[factory]
        params = {p["name"]: p["default"] for p in entry["params"]} | parameters
        entries.append((factory, parameters, params, entry))
    return entries


def _size(path):
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def _trim(root, current):
    budget = memory_budget("pool_disk", 2048)
    directories = [p for p in root.iterdir() if p.is_dir()]
    sizes = {p: _size(p) for p in directories}
    total = sum(sizes.values())
    for directory in sorted(directories, key=lambda p: p.stat().st_mtime_ns):
        if total <= budget:
            break
        if directory != current:
            shutil.rmtree(directory)
            total -= sizes[directory]
    if sizes.get(current, 0) > budget:
        shutil.rmtree(current)
        raise RuntimeError("pool exceeds MILL_POOL_DISK_MIB; increase the disk cache budget")


def _assemble(directory, raw, entries):
    fields = {name: [] for name in ("names", "natives", "factories", "sources", "params", "skipped")}
    columns = []
    for i, (factory, given, params, entry) in enumerate(entries):
        if (directory / f"{i}.skip").exists():
            fields["skipped"].append(factory)
            continue
        part = pl.read_ipc(directory / f"{i}.arrow", memory_map=True)
        if part.width and part.height != raw.height:
            raise ValueError("persisted factory output has the wrong row count")
        for column, dtype in part.schema.items():
            if not (dtype.is_numeric() or dtype == pl.Boolean):
                continue
            fields["names"].append(column if column not in fields["names"] else f"{factory}:{column}")
            fields["natives"].append(column.split("__", 1)[0] if given else column)
            fields["factories"].append(factory)
            fields["sources"].append(entry["source"])
            fields["params"].append(params)
            columns.append((i, column))
    with _atomic_path(directory / "values.npy") as temporary:
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64,
                    shape=(raw.height, len(columns)), fortran_order=True)
        last, part = None, None
        for j, (i, column) in enumerate(columns):
            if i != last:
                part = pl.read_ipc(directory / f"{i}.arrow", memory_map=True)
                last = i
            array = part[column].cast(pl.Float64).to_numpy().copy()
            array[~np.isfinite(array)] = np.nan
            values[:, j] = array
        values.flush()
        del values
    with _atomic_path(directory / "ms.npy") as temporary:
        with temporary.open("wb") as stream:
            np.save(stream, raw["datetime"].dt.epoch("ms").to_numpy(), allow_pickle=False)
    write_json_atomic(directory / "ready.json", fields)
    # The completed map replaces its resumable per-factory checkpoints.
    for path in (*directory.glob("*.arrow"), *directory.glob("*.skip")):
        path.unlink(missing_ok=True)


def pool_for(raw, bar, spec, digest, *, allow_stale=True, quantum=5):
    root = cache_root() / "pools"
    root.mkdir(parents=True, exist_ok=True)
    family = _hash([_revision(), bar, spec])
    directory = root / f"{family}-{digest}"
    if (directory / "ready.json").exists():
        return _load(directory), True, 0.0
    older = sorted((p for p in root.glob(f"{family}-*") if (p / "ready.json").exists()),
                   key=lambda p: p.stat().st_mtime_ns, reverse=True)
    directory.mkdir(exist_ok=True)
    began = time.perf_counter()
    entries = _entries(spec)
    for i, (factory, given, _, entry) in enumerate(entries):
        path = directory / f"{i}.arrow"
        skipped = directory / f"{i}.skip"
        if not path.exists() and not skipped.exists():
            try:
                part = entry["producer"](raw, **given)
            except Exception:
                write_json_atomic(skipped, {"factory": factory})
            else:
                with _atomic_path(path) as temporary:
                    part.write_ipc(temporary, compression="uncompressed")
            _trim(root, directory)
            if time.perf_counter() - began >= quantum and i + 1 < len(entries):
                if allow_stale and older and older[0].exists():
                    return _load(older[0]), True, (time.perf_counter() - began) * 1000
                raise PoolPending("catalog construction is progressing; retry shortly")
    _assemble(directory, raw, entries)
    _trim(root, directory)
    return _load(directory), False, (time.perf_counter() - began) * 1000
