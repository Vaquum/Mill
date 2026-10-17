from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Mill"
APP_SLUG = "mill"
DEFAULT_BAR = "1h_time_bars"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_ARROW_DIR = "/opt/arrow"
ARROW_LATEST_NAME = "latest.arrow"


def memory_budget(name: str, default_mib: int) -> int:
    value = int(os.environ.get(f"MILL_{name.upper()}_MIB", default_mib))
    if value < 0:
        raise ValueError(f"MILL_{name.upper()}_MIB must be nonnegative")
    return value * 1024**2


def cache_root() -> Path:
    override = os.environ.get("MILL_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / APP_SLUG


def arrow_root() -> Path:
    override = os.environ.get("MILL_ARROW_DIR")
    return Path(override) if override else Path(DEFAULT_ARROW_DIR)


def arrow_latest_path(series: str) -> Path:
    return arrow_root() / series / ARROW_LATEST_NAME


def config_root() -> Path:
    override = os.environ.get("MILL_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".config" / APP_SLUG


def host() -> str:
    return os.environ.get("MILL_HOST", DEFAULT_HOST)


def port() -> int:
    raw = os.environ.get("MILL_PORT")
    return int(raw) if raw else DEFAULT_PORT
