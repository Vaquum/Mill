import importlib.util
from pathlib import Path
import signal
import subprocess
import sys


def test_a_signaled_regression_group_cannot_be_hidden_by_passing_groups(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "profiling" / "check.py"
    spec = importlib.util.spec_from_file_location("mill_check", path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    real = subprocess.Popen
    children = []
    def group(argv, **kwargs):
        script = ("import os,signal; os.kill(os.getpid(), signal.SIGTERM)" if not children else "pass")
        child = real([sys.executable, "-c", script], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(check.subprocess, "Popen", group)
    monkeypatch.setattr(sys, "argv", [str(path)])
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        assert check.main() != 0
        assert children[0].returncode == -signal.SIGTERM
        assert all(child.returncode == 0 for child in children[1:])
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
