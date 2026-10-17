"""Witness exact service outputs, allowing only the declared dependency version."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from engine_sweep import content_digest


def witness(value):
    """Rewrite only the provenance field of each exported rule, never its values."""
    paths = []
    normalized = copy.deepcopy(value)
    for index, candidate in enumerate(normalized["candidates"]):
        rule = candidate.get("rule")
        manifest = rule.get("manifest") if rule else None
        if manifest is not None:
            metadata = manifest["metadata"]
            assert metadata["limen_version"] == "5.29.2"
            paths.append(f"candidates/{index}/rule/manifest/metadata/limen_version")
            metadata["limen_version"] = "3.30.0"
    return {"paths": paths, "before": content_digest(normalized), "after": content_digest(value)}


def worker(args):
    import engine_sweep
    observations = []
    def capture(value):
        if isinstance(value, dict) and {"candidates", "context", "target", "pool"} <= value.keys():
            observations.append(witness(value))
        return content_digest(value)
    engine_sweep.content_digest = capture
    if args.kind == "view":
        driver, phases = engine_sweep, ("cold", "repeat", "display", "windows")
        extra = ["--case", "research-discover", "--k", "30", "--reps", "1", "--bar", "15m_time_bars"]
    else:
        import catalog_service
        driver, phases = catalog_service, ("cold",)
        extra = ["--bar", "1h_time_bars"]
    record = args.scratch/"run.json"
    sys.argv = [driver.__file__, *extra, "--parquet", str(args.parquet), "--output", str(record),
                "--worker", "--scratch", str(args.scratch)]
    assert driver.main() == 0
    assert len(observations) == len(phases)
    args.output.write_text(json.dumps({"run": json.loads(record.read_text()),
        "version_field": "candidates/*/rule/manifest/metadata/limen_version",
        "before_version": "3.30.0", "after_version": "5.29.2",
        "observations": dict(zip(phases, observations))}, indent=2)+"\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("view", "catalog"), required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    def interrupted(signum, frame):
        raise InterruptedError(f"output comparison interrupted: {signum}")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with tempfile.TemporaryDirectory(prefix="mill-output-proof-") as temporary:
        child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--worker", "--scratch", temporary],
                                 start_new_session=True)
        try:
            code = child.wait(timeout=50)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=1)
        assert code == 0, "service output witness failed"
    print(f"Captured exact {args.kind} outputs and the dependency-version-only witness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
