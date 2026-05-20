"""Read BTCUSDT klines parquet snapshots from Hugging Face datasets.

Each bar size is hosted as its own dataset at vaquum/binance_btcusdt_<size>_klines.
The latest.json sidecar declares the current parquet file name plus a sha256
for integrity. Schema matches the ClickHouse export exactly (17 columns).
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request

import polars as pl

from frames import normalize_datetime


_BASE = "https://huggingface.co/datasets/vaquum/binance_btcusdt_{size}_klines/resolve/main"
_BAR_TO_SIZE = {
    "15m_time_bars": "15m",
    "30m_time_bars": "30m",
    "1h_time_bars":  "1h",
    "2h_time_bars":  "2h",
    "4h_time_bars":  "4h",
}


def _http_get(url: str, *, timeout: int) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def klines(bar_name: str) -> pl.DataFrame:
    if bar_name not in _BAR_TO_SIZE:
        raise ValueError(f"Unsupported bar for HF source: {bar_name!r}")
    base = _BASE.format(size=_BAR_TO_SIZE[bar_name])
    meta = json.loads(_http_get(f"{base}/latest.json", timeout=30).decode())
    file_name = meta["file_name"]
    expected = meta["sha256"]
    payload = _http_get(f"{base}/{file_name}", timeout=180)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"sha256 mismatch for {bar_name} ({file_name}): "
            f"expected {expected}, got {actual}"
        )
    df = pl.read_parquet(io.BytesIO(payload))
    return normalize_datetime(df)
