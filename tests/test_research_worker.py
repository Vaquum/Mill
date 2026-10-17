import os
import subprocess
import sys

import polars as pl
import pytest

from mill import qc
from mill.research import Worker, ResearchBusy, ResearchTimeout


def test_real_worker_matches_report_and_runs_outside_server_process():
    frame = pl.DataFrame({"a": [1.0, None, 2.0, 2.0]})
    worker = Worker(timeout=15)
    try:
        assert worker.run("report", frame=frame, target=None) == qc.report(frame, None)
        assert worker._process.pid != os.getpid()
        pid = worker._process.pid
        assert worker.run("report", frame=frame, target=None) == qc.report(frame, None)
        assert worker._process.pid == pid
    finally:
        worker.close()
    assert worker._process is None


def test_worker_deadline_reaps_process_and_can_restart():
    worker = Worker(timeout=0.1)
    process = subprocess.Popen([sys.executable, "-c", "import time; input(); time.sleep(30)"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    worker._process = process
    try:
        with pytest.raises(ResearchTimeout):
            worker.run("report", target=None)
        assert process.poll() is not None and worker._process is None
        worker.timeout = 15
        assert worker.run("report", frame=pl.DataFrame({"a": [1.0]}), target=None)["n_rows"] == 1
    finally:
        worker.close()


def test_worker_admits_only_one_active_and_one_waiting_request():
    worker = Worker()
    assert worker._slots.acquire(False) and worker._slots.acquire(False)
    try:
        with pytest.raises(ResearchBusy):
            worker.run("report", target=None)
    finally:
        worker._slots.release()
        worker._slots.release()
        worker.close()
