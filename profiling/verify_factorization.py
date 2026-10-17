"""Verify measured performance evidence without rerunning hardware benchmarks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

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
    drivers = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ("profiling/engine_sweep.py", "profiling/catalog_service.py")}
    repetitions = Counter()
    groups = {}

    def read(path, side, ablation="none"):
        sample = json.loads((root / path).read_text())
        assert sample["status"] == "complete", path
        assert sample["ablation"] == ablation, path
        driver = sample.get("profiler_path", "profiling/engine_sweep.py")
        assert driver in drivers and sample["profiler_sha256"] == drivers[driver], path
        if driver == "profiling/catalog_service.py":
            assert sample["common_profiler_sha256"] == drivers["profiling/engine_sweep.py"], path
        fingerprint = document["reference_source_sha256"] if side == "before" else document["optimized_source_sha256"]
        assert sample["environment"]["source_tree_sha256"] == fingerprint, path
        assert sample["input"]["rows"] == sample["input"]["source_rows"], "full source history is required"
        assert sample["peak_rss_bytes"] > 0
        for measurement in sample["measurements"]:
            assert measurement["samples_ms"] and all(value >= 0 for value in measurement["samples_ms"])
        return sample

    def matched(left, right, parity):
        assert left["profiler_sha256"] == right["profiler_sha256"]
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
        groups.setdefault(pair["group"], []).append((before, after))
    assert repetitions and all(count >= 3 for count in repetitions.values()), "three matched process pairs per workload"
    assert set(repetitions) >= {"workload-83k", "qc-20k", "qc-83k", "discover-full-83k",
        "discover-20k", "transition-20k", "pool-83k", "research-discover-83k", "http-83k"}
    for group, observations in groups.items():
        for name, summary in document["summaries"][group].items():
            if "wall_speedup" not in summary:
                continue
            for side, index in (("before", 0), ("after", 1)):
                samples = [next(m for m in pair[index]["measurements"] if m["name"] == name)["median_ms"]
                           for pair in observations]
                assert summary[f"{side}_wall_ms"] == {"median": statistics.median(samples),
                    "min": min(samples), "max": max(samples), "samples": samples}
            assert summary["wall_speedup"] == summary["before_wall_ms"]["median"]/summary["after_wall_ms"]["median"]
    assert {entry["component"] for entry in document["ablations"]} == {
        "native", "json", "pairs", "edges", "bootstrap", "null", "compact", "auc", "rules", "reuse", "context"}
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
