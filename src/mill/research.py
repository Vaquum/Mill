"""One bounded research process; the HTTP process retains data-serving capacity."""
from __future__ import annotations

import atexit
import contextlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time


class ResearchBusy(RuntimeError):
    pass


class ResearchTimeout(RuntimeError):
    pass


class Worker:
    def __init__(self, timeout: float = 45) -> None:
        self.timeout = timeout
        self._slots = threading.BoundedSemaphore(2)
        self._lock = threading.Lock()
        self._process = None
        atexit.register(self.close)

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            process.stdin.close()
            process.stdout.close()

    def run(self, kind: str, *, frame=None, **arguments):
        if not self._slots.acquire(blocking=False):
            raise ResearchBusy("research capacity is occupied; retry shortly")
        deadline = time.monotonic() + self.timeout
        acquired = False
        try:
            acquired = self._lock.acquire(timeout=max(0, deadline - time.monotonic()))
            if not acquired:
                raise ResearchTimeout("research queue deadline exceeded")
            with tempfile.TemporaryDirectory(prefix="mill-research-",
                                              dir=os.environ.get("MILL_RESEARCH_TMPDIR")) as directory:
                root = Path(directory)
                if frame is not None:
                    frame.write_ipc(root / "frame.arrow", compression="uncompressed")
                    arguments["frame"] = str(root / "frame.arrow")
                request = root / "request.json"
                request.write_text(json.dumps({"kind": kind, **arguments}))
                if self._process is None or self._process.poll() is not None:
                    self.close()
                    env = dict(os.environ)
                    threads = max(1, min(8, int(env.get("MILL_RESEARCH_THREADS", "4"))))
                    env.update({name: str(threads) for name in (
                        "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")})
                    self._process = subprocess.Popen(
                        [sys.executable, "-m", "mill.research"], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, env=env,
                    )
                process = self._process
                try:
                    process.stdin.write((str(request) + "\n").encode())
                    process.stdin.flush()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                        raise ResearchTimeout("research execution deadline exceeded")
                    if process.stdout.readline() != b"done\n":
                        raise ResearchTimeout("research worker exited before returning a result")
                    result = json.loads((root / "result.json").read_text())
                except (OSError, ResearchTimeout):
                    self.close()
                    raise
                if "error" in result:
                    if result["error"] == "PoolPending":
                        raise ResearchBusy(result["detail"])
                    if result["error"] in ("ValueError", "TypeError", "KeyError"):
                        raise ValueError(result["detail"])
                    raise RuntimeError(result["detail"])
                return result["value"]
        finally:
            if acquired:
                self._lock.release()
            self._slots.release()


_WORKER_ENV = None


def _execute(request):
    if request["kind"] == "report":
        import polars as pl
        from . import qc
        return qc.report(pl.read_ipc(request["frame"], memory_map=True), request["target"])
    from . import serve
    serve._PERSIST_POOLS = True
    global _WORKER_ENV
    environment = request["environment"]
    if environment != _WORKER_ENV:
        from .get_data import _DATA
        from .table import _PRODUCERS
        for cache in (serve._RAW_CACHE, serve._RESULTS, serve._QC_CACHE, serve._POOLS, _DATA, _PRODUCERS):
            cache.clear()
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _WORKER_ENV = environment
    serve._VIEW_SPECS = request["views"]
    if request["kind"] == "warm":
        from .pool_store import PoolPending, pool_for
        raw, _ = serve._raw_for(request["bar"])
        try:
            pool_for(raw, request["bar"], None, serve._RAW_DIGESTS[request["bar"]], allow_stale=False)
            return {"complete": True}
        except PoolPending:
            return {"complete": False}
    if request["kind"] == "discover":
        return serve._discover(request["body"])
    raise ValueError("unknown research operation")


def main():
    # Library diagnostics stay on stderr; stdout is a private completion channel.
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            path = Path(line.strip())
            signal.alarm(50)
            try:
                result = {"value": _execute(json.loads(path.read_text()))}
                payload = json.dumps(result, allow_nan=False)
            except Exception as exc:
                result = {"error": type(exc).__name__, "detail": str(exc)}
                payload = json.dumps(result)
            signal.alarm(0)
            path.with_name("result.json").write_text(payload)
            output.write("done\n")
            output.flush()


if __name__ == "__main__":
    main()
