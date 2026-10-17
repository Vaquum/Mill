"""Compare the full factory contract against the pinned reference on this runtime."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from prepare_limen_reference import ARCHIVE, DEFAULT, REFERENCE, verify

CURRENT = "728a6eefc19cde2cb0fa3f5312b185e2877bc330"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        signal.alarm(45)
        from limen_cases import snapshot
        args.output.write_text(json.dumps(snapshot(), sort_keys=True))
        return 0
    reference = args.reference.resolve()
    verify(reference)
    children = []
    def interrupted(signum, frame):
        raise InterruptedError(f"comparison interrupted: {signum}")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with tempfile.TemporaryDirectory(prefix="mill-limen-comparison-") as temporary:
        paths = [Path(temporary)/side for side in ("before", "after")]
        extra = os.environ.get("PYTHONPATH", "")
        source = str(Path(__file__).resolve().parents[1]/"src")
        try:
            for i,path in enumerate(paths):
                search = [str(reference), source, extra] if i == 0 else [source, extra]
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in search if p))
                children.append(subprocess.Popen([sys.executable, __file__, "--worker", "--output", str(path)],
                    env=env, start_new_session=True))
            deadline = time.monotonic()+48
            while any(child.poll() is None for child in children):
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("factory comparison", 48)
                time.sleep(.02)
            assert all(child.returncode == 0 for child in children), "comparison worker failed"
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=1)
        before, after = [json.loads(path.read_text()) for path in paths]
        assert before["origin"].get("url") == ARCHIVE
        assert after["origin"]["vcs_info"]["commit_id"] == CURRENT
        assert before["catalog"] == after["catalog"]
        for name,value in before["versions"].items():
            if name != "vaquum_limen":
                assert after["versions"][name] == value
        differences = []
        for case, left in before["cases"].items():
            right = after["cases"][case]
            assert left["rows"] == right["rows"] and left["input_digest"] == right["input_digest"]
            differences += [{"case": case, "factory": name, "before": value, "after": right["results"][name]}
                for name,value in left["results"].items() if value != right["results"][name]]
        from verify_factorization import tree
        root = Path(__file__).resolve().parents[1]
        result = {"reference": before, "current": after, "differences": differences,
                  "source_tree_sha256": tree(root),
                  "factory_contract_sha256": hashlib.sha256((root/"src/mill/factory_contract.json").read_bytes()).hexdigest(),
                  "reference_integrity": verify(reference),
                  "checks": len(before["catalog"])*len(before["cases"])}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        verify(reference)
        assert not differences, f"{len(differences)} factory/case mismatches; see {args.output}"
        print(f"Exact equivalence: {result['checks']} factory/case comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
