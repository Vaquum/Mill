"""Install the immutable numerical reference once, outside test execution."""
import hashlib
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile

REFERENCE = "86968e44022cabcae83fc38fe2ed0c7105816fa2"
DEFAULT = Path(__file__).resolve().parents[1]/".limen-reference"


def fingerprint(root):
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((root/"limen").rglob("*"))
              if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def verify(root):
    marker = json.loads((root/".mill-reference.json").read_text())
    assert marker["commit"] == REFERENCE
    assert marker["source_sha256"] == fingerprint(root), "reference package changed"
    return marker


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.directory.resolve()
    if root.exists():
        print(json.dumps(verify(root)))
        return 0
    root.parent.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise InterruptedError(f"reference preparation interrupted: {signum}")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with tempfile.TemporaryDirectory(prefix=".mill-reference-", dir=root.parent) as temporary:
        staging = Path(temporary)/"package"
        child = subprocess.Popen([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-deps", "--target", str(staging),
            f"git+https://github.com/Vaquum/Limen.git@{REFERENCE}"], start_new_session=True)
        try:
            code = child.wait(timeout=48)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=1)
        if code:
            return code
        origins = [json.loads(path.read_text()) for path in staging.glob("*.dist-info/direct_url.json")]
        assert any(item.get("vcs_info", {}).get("commit_id") == REFERENCE for item in origins)
        marker = {"commit": REFERENCE, "source_sha256": fingerprint(staging)}
        (staging/".mill-reference.json").write_text(json.dumps(marker))
        staging.rename(root)
    print(json.dumps(verify(root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
