"""Retained comparisons must describe identical inputs and exact outputs."""
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1] / "profiling"


def test_matched_performance_evidence():
    records = {p.name: json.loads(p.read_text()) for p in (ROOT / "results" / "optimization").glob("*.json")}
    assert records
    for name, current in records.items():
        assert current["status"] == "complete", name
        for measurement in current["measurements"]:
            assert measurement["median_ms"] == statistics.median(measurement["samples_ms"]), name
        before_name = name.replace("after-", "before-", 1)
        if not name.startswith("after-") or before_name not in records:
            continue
        before = records[before_name]
        assert before["source"] == current["source"], name
        assert before["input"] == current["input"], name
        for key in ("versions", "thread_environment"):
            assert before["environment"][key] == current["environment"][key], name
        for key in current:
            if key.startswith("parity_"):
                assert before[key] == current[key], (name, key)
        if "http_during_qc" in current:
            assert before["http_during_qc"]["qc"]["report_digest"] == current["http_during_qc"]["qc"]["report_digest"]
    service = records["after-83k-service-pool.json"]
    assert service["mapped_values"] and service["partial_preparation_steps_ms"]
    assert service["parity_pool"] == records["before-83k-discover-search.json"]["parity_pool"]
    report = json.loads((ROOT / "optimization.json").read_text())
    for item in report["timings"]:
        for side in ("before", "after"):
            document = records[item[side + "_file"]]
            actual = next(m["median_ms"] for m in document["measurements"] if m["name"] == item["measurement"])
            assert item[side + "_ms"] == actual
