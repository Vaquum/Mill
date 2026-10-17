"""Reuse is admitted only for a captured, immutable pool version and exact rows."""
from dataclasses import replace
import os

import numpy as np
import pytest

from mill import discover, discovery_reuse


def _pool(directory):
    directory.mkdir(exist_ok=True)
    rng = np.random.default_rng(81)
    n = 401
    values = rng.normal(size=(n, 4))
    values[5::9, 0] += 5
    np.save(directory / "values.npy", np.asfortranarray(values))
    np.save(directory / "ms.npy", discover.parse_ms("2024-01-01") + np.arange(n)*3_600_000)
    fields = dict(names=("a", "b", "c", "d"), natives=("a", "b", "c", "d"),
                  factories=("close",)*4, sources=("raw",)*4, params=({},)*4, skipped=())
    return discover.mapped_pool(directory, fields), fields


def _options(pool):
    ms, step = pool.ms, 3_600_000
    return dict(step_ms=step, windows=[(int(ms[i]), int(ms[i])+step) for i in range(5, len(ms), 9)],
                mode="inside", lead_bars=4, context=(int(ms[0]), int(ms[-1])+step),
                display=(int(ms[0]), int(ms[-1])+step), top=3, max_points=100,
                bar_source={"method": "limen.data.HistoricalData.get_arrow_file", "params": {"file_path": "/opt/arrow/time_1h/latest.arrow"}},
                label="test")


def _answer(pool, options):
    result = discover.discover(pool, **options)
    return {k: v for k, v in result.items() if k != "timing_ms"}


def test_related_runs_share_only_their_invariant_stages(tmp_path, monkeypatch):
    discovery_reuse.clear()
    pool, _ = _pool(tmp_path)
    base = _options(pool)
    variants = [base, base | {"top": 1, "max_points": 80, "display": (int(pool.ms[100]), base["display"][1]), "label": "other"},
                base | {"windows": [(a+3_600_000, b+3_600_000) for a, b in base["windows"]]},
                base | {"context": (int(pool.ms[17]), base["context"][1])}]
    counts = {"rank": 0, "null": 0}
    with monkeypatch.context() as patch:
        for name, label in (("column_ranks", "rank"), ("shift_null", "null")):
            original = getattr(discover, name)
            def wrapped(*args, _fn=original, _label=label):
                counts[_label] += 1
                return _fn(*args)
            patch.setattr(discover, name, wrapped)
        actual = [_answer(pool, options) for options in variants]
        assert counts == {"rank": 2, "null": 3}
        # Returned containers cannot mutate retained analysis or another request.
        actual[0]["candidates"][0]["similar"].append("changed")
        again = _answer(pool, base)
        assert "changed" not in again["candidates"][0]["similar"]
        assert counts == {"rank": 2, "null": 3}
    actual[0] = again
    uncached = replace(pool, _identity=None)
    assert actual == [_answer(uncached, options) for options in variants]
    assert discovery_reuse._RANKS.bytes <= discovery_reuse._RANKS.budget
    assert discovery_reuse._ANALYSES.bytes <= discovery_reuse._ANALYSES.budget


def test_atomic_replacement_cannot_label_old_mapping_as_new_version(tmp_path):
    discovery_reuse.clear()
    old, fields = _pool(tmp_path)
    options = _options(old)
    old_key = discovery_reuse.pool_key(old)
    before = _answer(old, options)
    stamp = (tmp_path / "values.npy").stat().st_mtime_ns
    changed = old.values.copy()
    changed[:, 0] *= -1
    replacement = tmp_path / "replacement.npy"
    np.save(replacement, changed)
    os.utime(replacement, ns=(stamp, stamp))
    replacement.replace(tmp_path / "values.npy")
    new = discover.mapped_pool(tmp_path, fields)
    assert discovery_reuse.pool_key(old) == old_key
    assert discovery_reuse.pool_key(new) != old_key
    assert _answer(old, options) == before
    actual = _answer(new, options)
    assert actual != before
    assert actual == _answer(replace(new, _identity=None), options)


def test_mapping_replacement_during_open_is_detected(tmp_path, monkeypatch):
    pool, fields = _pool(tmp_path)
    replacement = tmp_path / "replacement.npy"
    np.save(replacement, -pool.values)
    original = np.load
    calls = []
    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            replacement.replace(tmp_path / "values.npy")
        return result
    monkeypatch.setattr(np, "load", load)
    with pytest.raises(RuntimeError, match="changed while opening"):
        discover.mapped_pool(tmp_path, fields)


def test_mutable_library_pools_are_never_reused(tmp_path):
    pool, _ = _pool(tmp_path)
    mutable = replace(pool, values=pool.values.copy(), _identity=None)
    assert discovery_reuse.pool_key(mutable) is None
    options = _options(mutable)
    before = _answer(mutable, options)
    mutable.values[:, 0] *= -1
    assert _answer(mutable, options) != before
