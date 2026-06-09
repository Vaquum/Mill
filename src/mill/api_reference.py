"""Machine-readable reference for the Mill HTTP API.

A single source of truth for the endpoint surface, served verbatim at
``GET /api-reference`` (public, no auth). The enum values (bars, targets, row
modes) are injected from the live route constants so the document can never
disagree with what the endpoints accept, and ``tests/test_api_reference.py``
pins the documented paths to the actual routes in ``serve.py`` so a new or
renamed endpoint that skips the reference fails the build.
"""

from __future__ import annotations

from typing import Any


def _param(
    name: str,
    type_: str,
    *,
    required: bool = False,
    default: Any = None,
    values: list[str] | None = None,
    description: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "type": type_, "required": required}
    if default is not None:
        out["default"] = default
    if values is not None:
        out["values"] = values
    if description:
        out["description"] = description
    return out


def build_reference(
    *,
    app_name: str,
    bars: list[str],
    bar_default: str,
    targets: list[str],
    row_modes: list[str],
    data_window: dict[str, str],
) -> dict[str, Any]:
    """Build the API reference document from the live route enums."""
    bars_param = _param(
        "bars", "enum", default=bar_default, values=bars,
        description="Bar series to load.",
    )
    target_param = _param(
        "target", "enum", default="none", values=targets,
        description="Target column to attach, or 'none'.",
    )
    features_param = _param(
        "features", "json", default="[]",
        description="JSON array of {factory, params} feature specs; factory "
                    "names come from /app-config.",
    )

    endpoints: list[dict[str, Any]] = [
        {
            "method": "GET", "path": "/health", "auth": False,
            "summary": "Liveness probe.",
            "response": {"content_type": "application/json",
                         "description": '{"status":"ok"}.'},
        },
        {
            "method": "GET", "path": "/api-reference", "auth": False,
            "summary": "This document: the machine-readable API reference.",
            "response": {"content_type": "application/json",
                         "description": "This reference object."},
        },
        {
            "method": "GET", "path": "/", "auth": True,
            "summary": "Web UI (single-page app).",
            "response": {"content_type": "text/html; charset=utf-8",
                         "description": "The Mill workbench HTML."},
        },
        {
            "method": "GET", "path": "/app-config", "auth": True,
            "summary": "UI configuration: bars, targets, factories, views, "
                       "bar data sources, and the data window.",
            "response": {"content_type": "application/json",
                         "description": "Configuration object consumed by the UI."},
        },
        {
            "method": "GET", "path": "/version", "auth": True,
            "summary": "Build provenance from the image environment.",
            "response": {"content_type": "application/json",
                         "description": '{"sha","date","subject"}; empty when '
                                        "unset."},
        },
        {
            "method": "GET", "path": "/data", "auth": True,
            "summary": "Build a feature frame and return it as JSON rows.",
            "query": [bars_param, target_param, features_param],
            "response": {"content_type": "application/json",
                         "description": "{columns, rows} of datetime + features "
                                        "[+ target]."},
        },
        {
            "method": "GET", "path": "/qc", "auth": True,
            "summary": "Quality-control report and grades for a feature set.",
            "query": [
                bars_param, target_param, features_param,
                _param("cols", "string", default="",
                       description="Comma-separated feature columns to include "
                                   "(default: all)."),
                _param("from", "string", default="",
                       description="Inclusive start date, YYYY-MM-DD."),
                _param("to", "string", default="",
                       description="Inclusive end date, YYYY-MM-DD."),
                _param("row_mode", "enum", default="bar", values=row_modes,
                       description="Row grouping applied before scoring."),
                _param("cut_n", "int", default=5,
                       description="Bucket count for cut/quantile/decile modes "
                                   "(min 2)."),
            ],
            "response": {"content_type": "application/json",
                         "description": "{columns, rows, n_rows, grades}."},
        },
        {
            "method": "POST", "path": "/save_view", "auth": True,
            "summary": "Create or update a saved view.",
            "body": {
                "content_type": "application/json",
                "fields": [
                    _param("name", "string", required=True,
                           description="lowercase letters/digits/underscores, "
                                       "starts with a letter, <= 41 chars."),
                    _param("spec", "json", required=True,
                           description="Array of {factory, params} feature specs."),
                    _param("target", "enum", default="none", values=targets),
                ],
            },
            "response": {"content_type": "application/json",
                         "description": "{ok, msg, name}."},
        },
        {
            "method": "POST", "path": "/delete_view", "auth": True,
            "summary": "Delete a saved view (built-in views cannot be deleted).",
            "body": {
                "content_type": "application/json",
                "fields": [_param("name", "string", required=True)],
            },
            "response": {"content_type": "application/json",
                         "description": "{ok, msg, name}."},
        },
        {
            "method": "POST", "path": "/save_tags", "auth": True,
            "summary": "Set or clear the tag list for a factory.",
            "body": {
                "content_type": "application/json",
                "fields": [
                    _param("factory", "string", required=True,
                           description="A known factory name (see /app-config)."),
                    _param("tags", "json", required=True,
                           description="Array of tag strings; an empty list clears."),
                ],
            },
            "response": {"content_type": "application/json",
                         "description": "{ok, tags}."},
        },
    ]

    return {
        "app": app_name,
        "description": f"{app_name} HTTP API. Every endpoint requires "
                       "authentication except /health and /api-reference.",
        "auth": {
            "schemes": ["Basic", "Bearer"],
            "public_paths": ["/health", "/api-reference"],
            "description": "HTTP Basic (username/password) or Bearer (API "
                           "secret); 401 with WWW-Authenticate on failure.",
        },
        "data_window": data_window,
        "endpoints": endpoints,
    }
