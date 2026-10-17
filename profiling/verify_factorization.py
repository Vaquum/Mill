"""Verify measured performance evidence without rerunning hardware benchmarks."""
from collections import Counter
import hashlib
import json
from pathlib import Path

REFERENCE = "847cd481e82b1000496db082e52acb4d9794c207"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def tree(root):
    source = root / "src/mill"
    return digest({str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(source.rglob("*.py"))})


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    document = json.loads((root / "profiling/factorization.json").read_text())
    assert document["reference_commit"] == REFERENCE
    assert document["optimized_source_sha256"] == tree(root), "benchmarks must cover the published engine"
    harness = hashlib.sha256((root / "profiling/engine_sweep.py").read_bytes()).hexdigest()
    repetitions = Counter()

    def read(path, side, ablation="none"):
        sample = json.loads((root / path).read_text())
        assert sample["status"] == "complete", path
        assert sample["ablation"] == ablation, path
        assert sample["profiler_sha256"] == harness, path
        fingerprint = document["reference_source_sha256"] if side == "before" else document["optimized_source_sha256"]
        assert sample["environment"]["source_tree_sha256"] == fingerprint, path
        assert sample["input"]["rows"] == sample["input"]["source_rows"], "full source history is required"
        assert sample["peak_rss_bytes"] > 0
        for measurement in sample["measurements"]:
            assert measurement["samples_ms"] and all(value >= 0 for value in measurement["samples_ms"])
        return sample

    def matched(left, right, parity):
        assert left["source"] == right["source"]
        assert left["input"] == right["input"]
        for name in ("versions", "limen_origin", "thread_environment", "polars_threads", "numba_threads", "numba_layer"):
            assert left["environment"][name] == right["environment"][name], name
        for name in parity:
            assert name in left and left[name] == right[name], name

    for pair in document["pairs"]:
        before, after = read(pair["before"], "before"), read(pair["after"], "after")
        matched(before, after, pair["parity"])
        repetitions[pair["group"]] += 1
    assert repetitions and all(count >= 3 for count in repetitions.values()), "three matched process pairs per workload"
    for ablation in document["ablations"]:
        assert len(ablation["runs"]) >= 3
        for run in ablation["runs"]:
            control = read(run["control"], "after")
            sample = read(run["sample"], "after", ablation["component"])
            matched(control, sample, run["parity"])
    return document


if __name__ == "__main__":
    evidence = verify()
    print(f"Verified {len(evidence['pairs'])} matched pairs and {len(evidence['ablations'])} component ablations")
