"""Bounded reuse for immutable, persisted discovery pools only."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import numpy as np

from .memo import ByteLRU, Flights
from .settings import memory_budget


def _size(value):
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(_size(k) + _size(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return size + sum(_size(v) for v in value)
    return size


_RANKS = ByteLRU(memory_budget("ranks", 256), lambda entry: entry[1])
_ANALYSES = ByteLRU(memory_budget("analyses", 16), lambda entry: entry[1])
_RULES = ByteLRU(memory_budget("rules", 16), lambda entry: entry[1])
_CONTEXTS = ByteLRU(memory_budget("contexts", 4), lambda entry: entry[1])
_FLIGHTS = Flights()


def clear():
    _RANKS.clear()
    _ANALYSES.clear()
    _RULES.clear()
    _CONTEXTS.clear()


def pool_key(pool):
    """Admit read-only file maps; mutable library arrays never enter reuse."""
    if pool._identity is None:
        return None
    shapes = []
    for array in (pool.values, pool.ms):
        filename = getattr(array, "filename", None)
        if filename is None or array.flags.writeable:
            return None
        shapes.append((array.shape, array.dtype.str))
    metadata = {name: getattr(pool, name) for name in
                ("names", "natives", "factories", "sources", "params", "skipped")}
    return pool._identity, tuple(shapes), hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).digest()


def ranks(key, build):
    if key is None:
        return build()
    with _FLIGHTS.hold(("ranks", key)):
        hit = _RANKS.get(key)
        if hit is None:
            value = build()
            value.setflags(write=False)
            hit = (value, value.nbytes + _size(key))
            _RANKS[key] = hit
        return hit[0]


def context_key(source_key, row_bits, pool, index):
    """A new source version need not change an already closed historical context.

    File identity admits a cheap first-level hit. Across source versions, hash
    the actual selected values before sharing numerical work. Never infer an
    unchanged context from append-only timestamps or the last row.
    """
    if source_key is None:
        return None
    key = source_key, row_bits
    with _FLIGHTS.hold(("context", key)):
        hit = _CONTEXTS.get(key)
        if hit is None:
            digest = hashlib.sha256()
            contiguous = index.size and index[-1]-index[0]+1 == index.size
            for j in range(pool.values.shape[1]):
                values = (pool.values[index[0]:index[-1]+1, j] if contiguous else pool.values[index, j])
                digest.update(memoryview(np.ascontiguousarray(values)))
            value = ((index.size, pool.values.shape[1]), pool.values.dtype.str, digest.digest(), source_key[-1])
            hit = value, _size(value) + _size(key)
            _CONTEXTS[key] = hit
        return hit[0]


def analysis(key, build):
    if key is None:
        return build()
    with _FLIGHTS.hold(("analysis", key)):
        hit = _ANALYSES.get(key)
        if hit is not None:
            return copy.deepcopy(hit[0]), {"rank": 0., "screen": 0., "null": 0., "stage_two": 0.}
        value, timing = build()
        _ANALYSES[key] = (copy.deepcopy(value), _size(value) + _size(key))
        return value, timing


def rule(key, build):
    if key is None:
        return build()
    with _FLIGHTS.hold(("rule", key)):
        hit = _RULES.get(key)
        if hit is None:
            value = build()
            if value is not None:
                value["_hits"].setflags(write=False)
            size = _size(key) + _size(value)
            if value is not None:
                size += value["_hits"].nbytes
            hit = value, size
            _RULES[key] = hit
        return hit[0].copy() if hit[0] is not None else None
