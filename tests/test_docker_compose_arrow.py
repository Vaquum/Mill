"""The deploy mounts the tdw Arrow bar store read-only at /opt/arrow.

Source-level checks on deploy/docker-compose.yml: the container cannot be
booted in a unit test, so the deploy invariants that let GetData read the store
are asserted against the compose file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = REPO_ROOT / "deploy" / "docker-compose.yml"


def _spec() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def _mill_mounts() -> list[object]:
    return _spec()["services"]["mill"]["volumes"]


def test_arrow_volume_mounted_read_only_at_opt_arrow() -> None:
    arrow = [
        m for m in _mill_mounts()
        if isinstance(m, dict) and m.get("target") == "/opt/arrow"
    ]
    assert len(arrow) == 1, arrow
    mount = arrow[0]
    assert mount["type"] == "volume"
    assert mount["read_only"] is True
    assert mount["source"] == "arrow"


def test_arrow_is_external_named_volume() -> None:
    arrow = _spec()["volumes"]["arrow"]
    assert arrow["external"] is True
    assert "tdw-control-plane_arrow" in str(arrow.get("name", ""))


def test_data_experiments_remains_first_rw_bind() -> None:
    first = _mill_mounts()[0]
    assert isinstance(first, dict)
    assert first["type"] == "bind"
    assert first["target"] == "/data/experiments"
    assert first.get("read_only") is not True


def test_arrow_dir_env_points_at_the_mount() -> None:
    env = _spec()["services"]["mill"]["environment"]
    assert env["MILL_ARROW_DIR"] == "/opt/arrow"
