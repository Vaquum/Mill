"""The /api-reference document stays honest: it matches the real routes, the
live enum constants, and the actual public/auth behavior of the server.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from mill import serve

_SERVE_SRC = Path(serve.__file__).read_text(encoding="utf-8")
_ENDPOINTS = {e["path"]: e for e in serve._API_REFERENCE["endpoints"]}


def test_reference_documents_exactly_the_routed_paths() -> None:
    # Every `url.path == "..."` in do_GET/do_POST is a real route; the document
    # must cover exactly those — no undocumented endpoint, no phantom entry.
    routed = set(re.findall(r'url\.path == "([^"]+)"', _SERVE_SRC))
    documented = set(_ENDPOINTS)
    assert documented == routed, {
        "routed_not_documented": routed - documented,
        "documented_not_routed": documented - routed,
    }


def test_enum_params_match_live_constants() -> None:
    qc_query = {p["name"]: p for p in _ENDPOINTS["/qc"]["query"]}
    assert qc_query["bars"]["values"] == list(serve._BARS)
    assert qc_query["bars"]["default"] == serve._BARS[0]
    assert qc_query["target"]["values"] == list(serve._TARGETS)
    assert qc_query["row_mode"]["values"] == sorted(serve._ROW_MODES)


def test_public_and_authed_flags_are_correct() -> None:
    assert _ENDPOINTS["/health"]["auth"] is False
    assert _ENDPOINTS["/api-reference"]["auth"] is False
    assert _ENDPOINTS["/app-config"]["auth"] is True
    assert _ENDPOINTS["/qc"]["auth"] is True
    assert set(serve._API_REFERENCE["auth"]["public_paths"]) == {
        "/health", "/api-reference",
    }


def test_every_endpoint_has_the_required_shape() -> None:
    for path, ep in _ENDPOINTS.items():
        assert ep["method"] in {"GET", "POST"}, path
        assert isinstance(ep["summary"], str) and ep["summary"], path
        assert ep["response"]["content_type"], path


@pytest.fixture
def server():
    serve._APP_USERNAME = "u"
    serve._APP_PASSWORD = "p"
    serve._API_SECRET = "s"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve._Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        thread.join(timeout=5)


def test_api_reference_served_publicly(server: str) -> None:
    with urllib.request.urlopen(f"{server}/api-reference") as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "application/json"
        doc = json.loads(resp.read())
    assert doc["app"] == serve.APP_NAME
    assert any(e["path"] == "/qc" for e in doc["endpoints"])


def test_health_public_but_app_config_requires_auth(server: str) -> None:
    with urllib.request.urlopen(f"{server}/health") as resp:
        assert resp.status == 200
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{server}/app-config")
    assert exc.value.code == 401
