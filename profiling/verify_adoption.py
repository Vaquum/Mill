"""Validate the current dependency-adoption evidence and its prior-stage link."""
import hashlib
import json
import sys
import statistics
from collections import Counter
from importlib.metadata import distribution

REFERENCE = "86968e44022cabcae83fc38fe2ed0c7105816fa2"
CURRENT = "728a6eefc19cde2cb0fa3f5312b185e2877bc330"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, fingerprint):
    document = json.loads((root/"profiling/adoption.json").read_text())
    assert document["optimized_source_sha256"] == fingerprint
    assert document["factory_contract_sha256"] == sha(root/"src/mill/factory_contract.json")
    assert document["previous_report_sha256"] == sha(root/"profiling/factorization.json")
    previous = json.loads((root/"profiling/factorization.json").read_text())
    assert document["previous_source_sha256"] == previous["optimized_source_sha256"]
    assert document["upstream_commit"] == CURRENT
    expected = REFERENCE if sys.version_info >= (3, 14) else CURRENT
    assert json.loads(distribution("vaquum_limen").read_text("direct_url.json"))["vcs_info"]["commit_id"] == expected
    functional = json.loads((root/document["functional_comparison"]).read_text())
    assert not functional["differences"] and functional["checks"] == 226*11
    assert functional["target_checks"] == 7*11
    assert functional["reference"]["targets"] == functional["current"]["targets"]
    assert functional["source_tree_sha256"] == fingerprint
    assert functional["factory_contract_sha256"] == document["factory_contract_sha256"]
    assert len(functional["reference"]["cases"]) == 11
    assert len(functional["reference"]["catalog"]) == 226
    assert functional["reference"]["catalog"] == functional["current"]["catalog"]
    assert functional["current"]["catalog"] == json.loads((root/"src/mill/factory_contract.json").read_text())["factories"]
    for name,left in functional["reference"]["cases"].items():
        assert left == functional["current"]["cases"][name]
    drivers = {path: sha(root/path) for path in
        ("profiling/engine_sweep.py", "profiling/catalog_service.py")}
    assert document["version_witness_driver_sha256"] == sha(root/"profiling/check_adoption_outputs.py")
    witnesses = {name: json.loads((root/path).read_text()) for name,path in document["version_witnesses"].items()}
    assert set(witnesses) == {"research-discover-83k", "catalog-service-20k"}
    for proof in witnesses.values():
        assert proof["before_version"] == "3.30.0" and proof["after_version"] == "5.29.2"
        assert proof["version_field"] == "candidates/*/rule/manifest/metadata/limen_version"
        run = proof["run"]
        assert run["status"] == "complete" and run["environment"]["source_tree_sha256"] == fingerprint
        driver = run.get("profiler_path", "profiling/engine_sweep.py")
        assert run["profiler_sha256"] == drivers[driver]
        if proof["setup_reuse"]:
            assert proof["setup_reuse"] == "prepared immutable pool; reconstruct the recorded cold cache flag"
            observed = proof["observations"]["cold"]
            assert observed["replayed_cached"] is True and observed["recorded_cached"] is False
            assert run["pool"]["cached"] is True and run["parity_research"] == observed["replayed_digest"]
    counts = Counter()
    groups = {}
    required_parity = {"workload-83k": ["parity_workload"],
        "pool-83k": ["parity_pool", "parity_pool_metadata"],
        "catalog-service-20k": ["parity_research"],
        "research-discover-83k": ["parity_research"],
        "http-83k": ["parity_http_data", "parity_http_qc"],
        "qc-83k": ["parity_report"]}
    for pair in document["pairs"]:
        before, after = [json.loads((root/pair[side]).read_text()) for side in ("before", "after")]
        for sample in (before, after):
            assert sample["status"] == "complete"
            driver = sample.get("profiler_path", "profiling/engine_sweep.py")
            assert sample["profiler_sha256"] == drivers[driver]
            if driver == "profiling/catalog_service.py":
                assert sample["common_profiler_sha256"] == drivers["profiling/engine_sweep.py"]
            assert sample["ablation"] == "none"
            assert sample["environment"]["source_tree_sha256"] == fingerprint
            assert sample["environment"]["engine_sha"] == document["measured_commit"]
            assert sample["input"]["rows"] == sample["input"]["source_rows"]
            assert sample["peak_rss_bytes"] > 0
            for measurement in sample["measurements"]:
                assert measurement["median_ms"] == statistics.median(measurement["samples_ms"])
                assert all(value >= 0 for value in measurement["samples_ms"])
                assert all(value >= 0 for value in measurement["process_cpu_samples_ms"])
        assert before["profiler_sha256"] == after["profiler_sha256"]
        assert before["source"] == after["source"] and before["input"] == after["input"]
        assert pair["parity"] == required_parity[pair["group"]]
        name, size = pair["group"].rsplit("-", 1)
        assert before["case"] == name
        assert before["input"]["rows"] == (20784 if size == "20k" else 83136)
        for name in pair["parity"]:
            assert name in before and name in after
            if pair["group"] in witnesses:
                proof = witnesses[pair["group"]]
                assert proof["run"]["source"] == after["source"] and proof["run"]["input"] == after["input"]
                if proof["setup_reuse"]:
                    assert before["pool"]["cached"] is False and after["pool"]["cached"] is False
                for side, sample in (("before", before), ("after", after)):
                    expected = {phase: value[side] for phase,value in proof["observations"].items()}
                    if pair["group"] == "catalog-service-20k":
                        expected = expected["cold"]
                    assert sample[name] == expected, (name, side)
            else:
                assert before[name] == after[name], name
        for name in ("thread_environment", "polars_threads", "numba_threads", "numba_layer"):
            assert before["environment"][name] == after["environment"][name]
        for name, value in before["environment"]["versions"].items():
            if name != "vaquum_limen":
                assert after["environment"]["versions"][name] == value
        # The legacy control is installed from an immutable source archive.
        origin = before["environment"]["limen_origin"]
        assert origin.get("vcs_info", {}).get("commit_id") == REFERENCE or origin["url"] == f"https://github.com/Vaquum/Limen/archive/{REFERENCE}.tar.gz"
        assert after["environment"]["limen_origin"]["vcs_info"]["commit_id"] == CURRENT
        counts[pair["group"]] += 1
        groups.setdefault(pair["group"], []).append((before, after))
    assert set(counts) == {"workload-83k", "pool-83k", "catalog-service-20k",
        "research-discover-83k", "http-83k", "qc-83k"}
    assert all(count >= 3 for count in counts.values())
    def summary(values):
        return {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}
    for group, observations in groups.items():
        for name, entry in document["summaries"][group].items():
            for index, side in enumerate(("before", "after")):
                if "wall_speedup" in entry:
                    values = [next(m for m in p[index]["measurements"] if m["name"] == name) for p in observations]
                    assert entry[side+"_wall_ms"] == summary([m["median_ms"] for m in values])
                    assert entry[side+"_process_cpu_ms"] == summary([statistics.median(m["process_cpu_samples_ms"]) for m in values])
                else:
                    assert entry[side] == summary([p[index][name] for p in observations])
            if "wall_speedup" in entry:
                assert entry["wall_speedup"] == entry["before_wall_ms"]["median"]/entry["after_wall_ms"]["median"]
    return document
