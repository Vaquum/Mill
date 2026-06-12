"""Concurrent launches of qc's one parallel=True kernel must not abort.

numba's workqueue threading layer kills the whole process (SIGABRT, exit 134)
on concurrent access; serve runs QC on a ThreadingHTTPServer, so two /qc
requests launch the bootstrap kernel from two threads. The hammer runs in a
child interpreter so a regression kills the child, not the pytest runner.

Hermetic: synthetic arrays, no data dependency. Arrays are distinct per call
so the content-addressed memo cannot short-circuit the launches.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HAMMER = """
import threading

import numpy as np

from mill import qc

rng = np.random.default_rng(0)
pairs = [
    (rng.normal(size=20_000), rng.normal(size=20_000)) for _ in range(8)
]


def run(i):
    x, y = pairs[i]
    for k in range(3):
        qc._bootstrap_spearman_ci(x + float(k), y, 8, seed=i, n_iter=100)


threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("ok")
"""


def test_concurrent_bootstrap_ci_does_not_abort() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [sys.executable, "-c", _HAMMER],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, (
        f"hammer child exited {proc.returncode} "
        f"(134 means the workqueue abort is back)\n{proc.stderr}"
    )
    assert proc.stdout.strip() == "ok"
