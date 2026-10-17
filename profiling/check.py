"""Run repository checks with a hard deadline below one minute."""
import os
from pathlib import Path
import signal
import subprocess
import sys


def interrupted(signum, frame):
    raise InterruptedError(f"check supervisor received signal {signum}")


def main():
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    child = subprocess.Popen([sys.executable, "-m", "pytest", "-q", *sys.argv[1:]],
                             cwd=Path(__file__).resolve().parents[1], start_new_session=True)
    try:
        return child.wait(timeout=50)
    except BaseException as exc:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        if isinstance(exc, subprocess.TimeoutExpired):
            return 124
        raise


if __name__ == "__main__":
    raise SystemExit(main())
