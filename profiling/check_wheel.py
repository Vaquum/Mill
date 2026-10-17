"""Build the published package and verify the runtime contract is included."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    def interrupted(signum, frame):
        raise InterruptedError(f"wheel check interrupted: {signum}")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with tempfile.TemporaryDirectory(prefix="mill-wheel-") as directory:
        child = subprocess.Popen([sys.executable, "-m", "pip", "wheel", "--no-deps",
            "--disable-pip-version-check", "--wheel-dir", directory, str(root)], start_new_session=True)
        try:
            code = child.wait(timeout=48)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=1)
        assert code == 0, "wheel build failed"
        wheels = list(Path(directory).glob("*.whl"))
        assert len(wheels) == 1
        with zipfile.ZipFile(wheels[0]) as wheel:
            assert wheel.read("mill/factory_contract.json") == (root/"src/mill/factory_contract.json").read_bytes()
            assert "mill/py.typed" in wheel.namelist()
            assert "mill/assets/index.html" in wheel.namelist()
    print("Built wheel includes the exact factory contract and existing runtime assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
