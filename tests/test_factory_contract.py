"""The dependency can improve kernels without expanding Mill's public API."""
import hashlib
import copy
import json
from pathlib import Path
import tomllib
from packaging.requirements import Requirement
import pytest

from mill import feature_registry, pool_store


@pytest.mark.parametrize("version,commit", [("3.12", "728a6eefc19cde2cb0fa3f5312b185e2877bc330"),
    ("3.13", "728a6eefc19cde2cb0fa3f5312b185e2877bc330"),
    ("3.14", "86968e44022cabcae83fc38fe2ed0c7105816fa2")])
def test_dependency_upgrade_preserves_existing_interpreter_admission(version, commit):
    project = tomllib.loads((Path(__file__).resolve().parents[1]/"pyproject.toml").read_text())["project"]
    requirements = [Requirement(value) for value in project["dependencies"]]
    selected = [r for r in requirements if r.name == "vaquum_limen"
                and (r.marker is None or r.marker.evaluate({"python_version": version}))]
    assert len(selected) == 1 and selected[0].url.endswith("@"+commit)
    if version < "3.14":
        assert selected[0].extras == {"data"}, "GetData requires Limen's Arrow data dependencies"


def test_existing_public_factory_catalog_is_unchanged():
    encoded = json.dumps(feature_registry.catalog(), sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == "bc49145290d3868f5fe36451ebe8edacca4d862645e5fefc27ec67a1dae55c3c"
    assert len(feature_registry.REGISTRY) == 226


def test_upstream_additions_do_not_silently_add_features(monkeypatch):
    monkeypatch.setattr(feature_registry._LF, "unrequested_factory", lambda frame: frame, raising=False)
    assert "unrequested_factory" not in feature_registry._registry()


def test_missing_supported_factory_fails_at_admission(monkeypatch):
    monkeypatch.delattr(feature_registry._LI, "ema")
    with pytest.raises(AttributeError, match="ema"):
        feature_registry._registry()


def test_metadata_consumers_cannot_mutate_the_registry():
    catalog = feature_registry.catalog()
    row = next(item for item in catalog if item["name"] == "cci")
    next(p for p in row["params"] if p["name"] == "period")["default"] = -1
    params = feature_registry.params_for("cci")
    next(p for p in params if p["name"] == "period")["default"] = -2
    assert next(p for p in feature_registry.REGISTRY["cci"]["params"] if p["name"] == "period")["default"] == 14


def test_loaded_contract_changes_invalidate_persisted_pools(monkeypatch):
    before = pool_store._revision()
    replacement = dict(feature_registry.REGISTRY["close"],
        params=({"name": "revision_probe", "type": "int", "default": 1},))
    monkeypatch.setitem(feature_registry.REGISTRY, "close", replacement)
    assert pool_store._revision() != before


def test_service_witness_keeps_numerical_bits_and_unrelated_metadata(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"profiling"))
    from check_adoption_outputs import content_digest, witness
    value = {"candidates": [{"rule": {"manifest": {
        "metadata": {"limen_version": "5.29.2"}, "threshold": 1.}}}],
        "limen_version": "unrelated field", "signed_zero": -0.}
    reference = copy.deepcopy(value)
    reference["candidates"][0]["rule"]["manifest"]["metadata"]["limen_version"] = "3.30.0"
    proof = witness(value)
    assert proof["before"] == content_digest(reference)
    assert proof["after"] == content_digest(value)
    assert proof["paths"] == ["candidates/0/rule/manifest/metadata/limen_version"]
    reference["candidates"][0]["rule"]["manifest"]["threshold"] = float.fromhex("0x1.0000000000001p+0")
    assert proof["before"] != content_digest(reference)
    assert value["candidates"][0]["rule"]["manifest"]["metadata"]["limen_version"] == "5.29.2"
