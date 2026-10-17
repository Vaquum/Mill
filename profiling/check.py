"""Run repository checks with a hard deadline below one minute."""
import os
from contextlib import suppress
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def interrupted(signum, frame):
    raise InterruptedError(f"check supervisor received signal {signum}")


def main():
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    process_tests = ["tests/test_qc_thread_safety.py", "tests/test_research_worker.py"]
    profiler_tests = ["tests/test_engine_sweep.py"]
    groups = [sys.argv[1:]] if len(sys.argv) > 1 else [
        ["tests", *(f"--ignore={p}" for p in process_tests + profiler_tests)],
        profiler_tests, process_tests,
    ]
    children = []
    with tempfile.TemporaryDirectory(prefix="mill-check-") as directory:
        logs = [Path(directory) / str(i) for i in range(len(groups))]
        try:
            for group, path in zip(groups, logs):
                with path.open("w") as stream:
                    children.append(subprocess.Popen([sys.executable, "-m", "pytest", "-q", *group],
                        cwd=Path(__file__).resolve().parents[1], stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True))
            deadline = time.monotonic() + 50
            while any(child.poll() is None for child in children):
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("regression groups", 50)
                time.sleep(0.02)
            return 1 if any(child.returncode != 0 for child in children) else 0
        except subprocess.TimeoutExpired:
            return 124
        finally:
            for child in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                with suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=1)
            for path in logs:
                if path.exists():
                    sys.stdout.write(path.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
