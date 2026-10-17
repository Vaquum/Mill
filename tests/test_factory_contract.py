"""The dependency can improve kernels without expanding Mill's public API."""
import hashlib
import json
import pytest

from mill import feature_registry, pool_store


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
