from __future__ import annotations

import os
from datetime import date
from pathlib import Path

APP_NAME = "Mill"
APP_SLUG = "mill"
PROJECT_START = date(2024, 1, 1)
PROJECT_END = date(2026, 5, 15)
PROJECT_START_STR = PROJECT_START.isoformat()
PROJECT_END_STR = PROJECT_END.isoformat()
DEFAULT_BAR = "1h_time_bars"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def cache_root() -> Path:
    override = os.environ.get("MILL_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / APP_SLUG


def config_root() -> Path:
    override = os.environ.get("MILL_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".config" / APP_SLUG


def host() -> str:
    return os.environ.get("MILL_HOST", DEFAULT_HOST)


def port() -> int:
    raw = os.environ.get("MILL_PORT")
    return int(raw) if raw else DEFAULT_PORT
