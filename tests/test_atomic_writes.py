"""Config and metadata JSON survive a crash mid-write.

views.json / feature_tags.json were written with plain write_text; a process
death mid-write left torn JSON that the tolerant loaders silently replaced
with {} — every saved view lost. write_json_atomic dumps to a sibling tmp
file and renames, so the previous valid file survives any failure; serve must
not write JSON directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mill import serve
from mill.cache_store import write_json_atomic


def test_roundtrip_and_failure_preserves_previous(tmp_path: Path) -> None:
    target = tmp_path / "views.json"
    first = {"alpha": {"spec": [], "target": "none"}}
    write_json_atomic(target, first)
    assert json.loads(target.read_text()) == first
    assert list(tmp_path.iterdir()) == [target]

    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})
    assert json.loads(target.read_text()) == first
    assert list(tmp_path.iterdir()) == [target]

    second = {"beta": {"spec": [], "target": "none"}}
    write_json_atomic(target, second)
    assert json.loads(target.read_text()) == second


def test_serve_source_has_no_write_text() -> None:
    assert "write_text" not in Path(serve.__file__).read_text(encoding="utf-8")
