"""Validate the current dependency-adoption evidence and its prior-stage link."""
import hashlib
import json
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
    assert json.loads(distribution("vaquum_limen").read_text("direct_url.json"))["vcs_info"]["commit_id"] == CURRENT
    functional = json.loads((root/document["functional_comparison"]).read_text())
    assert not functional["differences"] and functional["checks"] == 226*11
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
    counts = Counter()
    for pair in document["pairs"]:
        before, after = [json.loads((root/pair[side]).read_text()) for side in ("before", "after")]
        for sample in (before, after):
            assert sample["status"] == "complete"
            driver = sample.get("profiler_path", "profiling/engine_sweep.py")
            assert sample["profiler_sha256"] == drivers[driver]
            assert sample["environment"]["source_tree_sha256"] == fingerprint
            assert sample["environment"]["engine_sha"] == document["measured_commit"]
            assert sample["input"]["rows"] == sample["input"]["source_rows"]
            assert sample["peak_rss_bytes"] > 0
        assert before["profiler_sha256"] == after["profiler_sha256"]
        assert before["source"] == after["source"] and before["input"] == after["input"]
        for name in pair["parity"]:
            assert name in before and before[name] == after[name], name
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
    assert set(counts) == {"workload-83k", "pool-83k", "catalog-service-20k",
        "research-discover-83k", "http-83k", "qc-83k"}
    assert all(count >= 3 for count in counts.values())
    return document
