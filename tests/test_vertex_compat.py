"""Tests for the Vertex/Gemini schema-compatibility layer.

Gemini Enterprise drops any tool whose JSON Schema uses anyOf without a concrete
type, unresolved $ref/$defs, or an array without items. These tests prove the
sanitizer removes all three from the schemas the server advertises, using the
server's own live tool definitions.
"""

from __future__ import annotations

from mcp.types import ListToolsRequest

from mcp_server_odoo.server import create_fastmcp_app
from mcp_server_odoo.tools import register_tools
from mcp_server_odoo.vertex_compat import (
    install_vertex_tool_sanitizer,
    sanitize_schema,
    vertex_compat_enabled,
)

# Keys Vertex/Gemini rejects anywhere in a tool schema.
FORBIDDEN_KEYS = ("anyOf", "oneOf", "allOf", "$ref", "$defs", "not", "definitions")


def _violations(node, path="<root>"):
    """Every place the schema still breaks Vertex rules: forbidden keys, a
    property without a type, or an array without items."""
    problems = []
    if isinstance(node, dict):
        for key in FORBIDDEN_KEYS:
            if key in node:
                problems.append(f"{path}: uses '{key}'")
        props = node.get("properties")
        if isinstance(props, dict):
            for name, sub in props.items():
                if isinstance(sub, dict) and "type" not in sub:
                    problems.append(f"{path}.properties.{name}: no 'type'")
                problems += _violations(sub, f"{path}.properties.{name}")
        if node.get("type") == "array" and "items" not in node:
            problems.append(f"{path}: array without 'items'")
        for key, value in node.items():
            if key != "properties":
                problems += _violations(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            problems += _violations(value, f"{path}[{i}]")
    return problems


# --- unit tests: sanitize_schema -------------------------------------------


def test_collapses_optional_anyof_to_nullable():
    schema = {
        "type": "object",
        "properties": {
            "order": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
    }
    out = sanitize_schema(schema)
    assert out["properties"]["order"] == {"type": "string", "nullable": True}
    assert not _violations(out)


def test_empty_any_branch_becomes_string():
    # Optional[Any] -> anyOf:[{}, {"type": "null"}]
    schema = {
        "type": "object",
        "properties": {"domain": {"anyOf": [{}, {"type": "null"}], "default": None}},
    }
    out = sanitize_schema(schema)
    assert out["properties"]["domain"]["type"] == "string"
    assert out["properties"]["domain"]["nullable"] is True


def test_resolves_ref_and_defs_inline():
    schema = {
        "type": "object",
        "properties": {"meta": {"$ref": "#/$defs/Meta"}},
        "$defs": {"Meta": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    }
    out = sanitize_schema(schema)
    assert "$defs" not in out
    assert out["properties"]["meta"]["properties"]["n"] == {"type": "integer"}
    assert not _violations(out)


def test_array_gets_items():
    schema = {"type": "object", "properties": {"tags": {"type": "array"}}}
    out = sanitize_schema(schema)
    assert out["properties"]["tags"]["items"] == {"type": "string"}


def test_drops_unsupported_keys():
    schema = {
        "type": "object",
        "title": "X",
        "additionalProperties": False,
        "properties": {"a": {"type": "string", "title": "A", "default": "z"}},
    }
    out = sanitize_schema(schema)
    assert "title" not in out
    assert "additionalProperties" not in out
    assert "default" not in out["properties"]["a"]


def test_recursive_ref_does_not_loop():
    schema = {
        "$ref": "#/$defs/Node",
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
    }
    out = sanitize_schema(schema)  # must terminate
    assert out["type"] == "object"


# --- gating -----------------------------------------------------------------


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("MCP_VERTEX_COMPAT", raising=False)
    assert vertex_compat_enabled() is True


def test_can_opt_out(monkeypatch):
    monkeypatch.setenv("MCP_VERTEX_COMPAT", "0")
    assert vertex_compat_enabled() is False


# --- integration: real server tools ----------------------------------------


def _build_app_with_tools():
    app = create_fastmcp_app()
    register_tools(app, connection=None, access_controller=None, config=None)
    return app


async def test_real_tools_are_dirty_before_sanitizing():
    """Guard: prove the raw schemas really do violate Vertex rules, so the
    integration test below is meaningful."""
    app = _build_app_with_tools()
    tools = await app.list_tools()
    dirty = [
        t.name for t in tools if _violations(t.inputSchema) or _violations(t.outputSchema or {})
    ]
    assert dirty, "expected raw schemas to be Vertex-incompatible"


async def test_every_tool_is_vertex_safe_after_sanitizing():
    app = _build_app_with_tools()
    tools = await app.list_tools()
    for tool in tools:
        problems = _violations(sanitize_schema(tool.inputSchema), tool.name)
        if tool.outputSchema:
            problems += _violations(sanitize_schema(tool.outputSchema), f"{tool.name}:out")
        assert not problems, f"{tool.name} still Vertex-incompatible: {problems}"


async def test_install_rewrites_tools_list_handler():
    """End-to-end: after install, the tools/list handler returns sanitized
    schemas for every tool."""
    app = _build_app_with_tools()
    install_vertex_tool_sanitizer(app)
    handler = app._mcp_server.request_handlers[ListToolsRequest]
    result = await handler(ListToolsRequest(method="tools/list"))
    tools = result.root.tools
    assert tools
    for tool in tools:
        assert not _violations(tool.inputSchema, tool.name)
        if tool.outputSchema:
            assert not _violations(tool.outputSchema, f"{tool.name}:out")
