"""Exercise the profiler's safety boundary through its public CLI."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import polars as pl
import pytest

from conftest import ArrowStore, ns

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "profiling" / "engine_sweep.py"


@pytest.fixture
def parquet(arrow_store: ArrowStore, tmp_path: Path) -> Path:
    frame = arrow_store.write("time_1h", [ns(2025, 1, 1) + i * 3_600_000_000_000 for i in range(120)])
    path = tmp_path / "input.parquet"
    frame.with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")).alias("datetime")).drop("ts").write_parquet(path)
    return path


def command(parquet, output, case="startup", *extra):
    return [sys.executable, str(SCRIPT), "--case", case, "--parquet", str(parquet), "--output", str(output), *extra]


def test_short_discovery_from_outside_checkout(parquet, tmp_path):
    output = tmp_path / "result.json"
    before = hashlib.sha256(parquet.read_bytes()).digest()
    subprocess.run(command(parquet, output, "discover", "--rows", "100", "--k", "1", "--reps", "1"),
                   cwd=tmp_path, check=True, timeout=45)
    result = json.loads(output.read_text())
    assert result["status"] == "complete" and result["input"]["rows"] == 100
    assert result["discovery"]["events"] > 0
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    assert result["environment"]["engine_sha"] == expected
    assert hashlib.sha256(parquet.read_bytes()).digest() == before


def test_input_is_never_the_output(parquet, tmp_path):
    before = parquet.read_bytes()
    result = subprocess.run(command(parquet, parquet), cwd=tmp_path, capture_output=True, timeout=5)
    assert result.returncode == 2 and parquet.read_bytes() == before


def test_worker_failure_is_recorded(parquet, tmp_path):
    output = tmp_path / "error.json"
    result = subprocess.run(command(parquet, output, "discover", "--rows", "3"),
                            cwd=tmp_path, capture_output=True, timeout=45)
    record = json.loads(output.read_text())
    assert result.returncode != 0 and record["status"] == "error"
    assert "at least four input bars" in record["error"]


def test_supervisor_sigterm_kills_worker_and_removes_fixtures(parquet, tmp_path):
    output = tmp_path / "result.json"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    process = subprocess.Popen(command(parquet, output), cwd=tmp_path,
                               env={**os.environ, "TMPDIR": str(scratch)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not output.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        worker = json.loads(output.read_text())["worker_pid"]
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=5)
        assert json.loads(output.read_text())["status"] == "interrupted"
        assert not list(scratch.glob("mill_engine_sweep_*"))
        with pytest.raises(ProcessLookupError):
            os.kill(worker, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_recorded_measurements_match_their_samples():
    records = list((ROOT / "profiling" / "results" / "2026-10-17").glob("*.json"))
    assert records
    for path in records:
        result = json.loads(path.read_text())
        assert result["status"] == "complete", path
        assert result["environment"]["versions"]["polars"] == "1.44.2"
        assert result["environment"]["limen_origin"]["vcs_info"]["commit_id"] == "86968e44022cabcae83fc38fe2ed0c7105816fa2"
        for measurement in result["measurements"]:
            assert all(value >= 0 for value in measurement["samples_ms"]), path
            assert measurement["median_ms"] == statistics.median(measurement["samples_ms"]), path
