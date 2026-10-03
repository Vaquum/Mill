"""The deploy publishes Mill on the host loopback only; Portal proxies it."""

from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "deploy" / "docker-compose.yml"


def test_port_published_on_host_loopback_only() -> None:
    spec = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert spec["services"]["mill"]["ports"] == ["127.0.0.1:${MILL_PORT:-8765}:8765"]
