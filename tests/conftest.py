"""Shared fixtures: a synthetic, hermetic tdw Arrow bar store.

The real store is written by the tdw-control-plane stack and mounted read-only
at ``/opt/arrow``. These fixtures write the same on-disk shape — a single
uncompressed Arrow IPC record batch with a strictly increasing ``Int64``-ns
``ts`` index — into a temp dir, and point ``MILL_ARROW_DIR`` / ``MILL_CACHE_DIR``
at temp locations so the tests never touch the host store or the real cache.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from mill.bars import BAR_VALUE_COLUMNS
from mill.settings import ARROW_LATEST_NAME


def ns(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """UTC wall-clock to Int64 nanoseconds — the store's ``ts`` encoding."""
    moment = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return int(moment.timestamp()) * 1_000_000_000


class ArrowStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(
        self,
        series: str,
        ts: list[int],
        *,
        extra: dict[str, pl.Series] | None = None,
    ) -> pl.DataFrame:
        """Write one series as a single-batch uncompressed Arrow IPC file.

        ``ts`` carries the index; every measure column is filled with a distinct
        ramp so a column mix-up is visible. ``extra`` adds dollar-only columns
        (``start_ts``, ``dollar_bar_id``) that Mill must drop.
        """
        height = len(ts)
        columns: dict[str, pl.Series] = {"ts": pl.Series(ts, dtype=pl.Int64)}
        for offset, name in enumerate(BAR_VALUE_COLUMNS):
            columns[name] = pl.Series(
                [float(offset * 100 + row) for row in range(height)],
                dtype=pl.Float64,
            )
        if extra:
            columns.update(extra)
        frame = pl.DataFrame(columns)
        path = self.root / series / ARROW_LATEST_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_ipc(
            path,
            compression="uncompressed",
            record_batch_size=max(frame.height, 1),
        )
        return frame


@pytest.fixture
def arrow_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ArrowStore:
    arrow_dir = tmp_path / "arrow"
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("MILL_ARROW_DIR", str(arrow_dir))
    monkeypatch.setenv("MILL_CACHE_DIR", str(cache_dir))
    return ArrowStore(arrow_dir)
