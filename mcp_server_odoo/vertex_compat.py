"""Vertex AI / Gemini schema-compatibility layer for the tools/list handshake.

Gemini Enterprise (Vertex AI function calling) parses tool JSON Schemas with a
strict subset of OpenAPI. It rejects a whole function declaration when a
parameter schema:

- uses ``anyOf`` / ``oneOf`` whose branches lack a concrete ``type`` (Pydantic v2
  renders every ``Optional[...]`` as ``anyOf:[S, {"type":"null"}]``, and
  ``Optional[Any]`` as ``anyOf:[{}, {"type":"null"}]`` — an empty ``{}`` with no
  type at all);
- carries unresolved ``$ref`` / ``$defs`` (Pydantic emits these for nested
  models used as tool return types);
- declares ``type: "array"`` without an ``items`` entry.

Anthropic/OpenAI parsers tolerate all of the above, so the server works from
Claude/ChatGPT but Gemini silently drops the tools and the model "describes a
plan" instead of calling them.

This module rewrites the announced schemas into the Vertex-safe subset.

``outputSchema`` is dropped rather than rewritten. The low-level MCP server
caches whatever ``tools/list`` returns and validates every ``tools/call``
result against that cached schema with ``jsonschema``, so the announced schema
is *not* advertising-only. The two consumers want incompatible things: Vertex
expresses "may be null" as OpenAPI's ``nullable: true`` and has no NULL type,
while ``jsonschema`` ignores ``nullable`` and enforces ``type``, so any
``Optional[...]`` field that is actually None fails validation
("None is not of type 'string'"). No single document satisfies both. Since an
``outputSchema`` is optional in MCP and Vertex needs the *input* schema to
build its function declaration, dropping it fixes the conflict outright;
``structuredContent`` is still returned to the client either way.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)

# Keys Vertex understands on a schema node. Everything else (title, default,
# additionalProperties, propertyNames, $schema, ...) is dropped.
_KEEP_KEYS = frozenset(
    {"type", "nullable", "description", "properties", "items", "enum", "required", "format"}
)

# `format` values Vertex accepts (per type). Anything else is dropped so it
# cannot trip validation.
_ALLOWED_FORMATS = frozenset({"date-time", "enum", "int32", "int64", "float", "double"})


def vertex_compat_enabled() -> bool:
    """Whether the sanitizer should run. Default ON; opt out with
    ``MCP_VERTEX_COMPAT`` set to a falsey value (0/false/no/off)."""
    return os.environ.get("MCP_VERTEX_COMPAT", "1").lower() not in ("0", "false", "no", "off")


def sanitize_schema(schema: Any) -> Any:
    """Return a deep copy of ``schema`` rewritten into the Vertex-safe subset.

    Resolves ``$ref``/``$defs`` inline, collapses ``anyOf``/``oneOf`` unions to a
    single typed branch (+ ``nullable``), guarantees every array has ``items``
    and every property node has a ``type``. Pure and side-effect free.
    """
    if not isinstance(schema, dict):
        return schema
    defs = schema.get("$defs") or schema.get("definitions") or {}
    resolved = _resolve_refs(copy.deepcopy(schema), defs, seen=())
    return _clean(resolved)


def _resolve_refs(node: Any, defs: dict, seen: tuple) -> Any:
    """Inline every ``$ref`` using ``defs``; break cycles defensively."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            name = ref.split("/")[-1]
            if name in seen:
                # Recursive model — stop and emit an opaque object.
                return {"type": "object"}
            target = defs.get(name)
            if target is None:
                return {"type": "object"}
            return _resolve_refs(copy.deepcopy(target), defs, seen + (name,))
        return {
            k: _resolve_refs(v, defs, seen)
            for k, v in node.items()
            if k not in ("$defs", "definitions")
        }
    if isinstance(node, list):
        return [_resolve_refs(v, defs, seen) for v in node]
    return node


def _collapse_union(subschemas: list) -> tuple[dict, bool]:
    """Collapse an anyOf/oneOf list into (chosen_schema, nullable)."""
    nullable = any(isinstance(s, dict) and s.get("type") == "null" for s in subschemas)
    candidates = [s for s in subschemas if not (isinstance(s, dict) and s.get("type") == "null")]
    chosen: dict | None = None
    # Prefer a branch that already carries a concrete type.
    for s in candidates:
        if isinstance(s, dict) and s.get("type"):
            chosen = s
            break
    if chosen is None:
        # All branches typeless (e.g. Optional[Any] -> {}). A JSON string is the
        # safe universal input: search_records/import_records already parse
        # domain/fields/context from a JSON string.
        chosen = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    return chosen, nullable


def _clean(node: Any) -> Any:
    if isinstance(node, list):
        return [_clean(v) for v in node]
    if not isinstance(node, dict):
        return node

    union = node.get("anyOf") or node.get("oneOf") or node.get("allOf")
    if isinstance(union, list) and union:
        chosen, nullable = _collapse_union(union)
        cleaned = _clean(chosen)
        if not isinstance(cleaned, dict):
            cleaned = {}
        # Carry a description down from the union parent if the branch lacks one.
        if node.get("description") and "description" not in cleaned:
            cleaned["description"] = node["description"]
        if nullable:
            cleaned["nullable"] = True
        return _finalize(cleaned)

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _KEEP_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: _clean(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = _clean(value)
        elif key == "format":
            if value in _ALLOWED_FORMATS:
                out[key] = value
        elif isinstance(value, (dict, list)):
            out[key] = _clean(value)
        else:
            out[key] = value
    return _finalize(out)


def _finalize(out: dict) -> dict:
    """Guarantee Vertex invariants: arrays have items, objects/props have a type."""
    if out.get("type") == "array" and "items" not in out:
        out["items"] = {"type": "string"}
    if "properties" in out and "type" not in out:
        out["type"] = "object"
    # Last resort: a property node Vertex would see with no type at all. Give it
    # one so the whole declaration is not rejected. Unions already resolved above.
    if "type" not in out and not any(k in out for k in ("properties", "items", "enum")):
        out["type"] = "string"
    return out


def sanitize_tool(tool):
    """Return a copy of an ``mcp.types.Tool`` with a Vertex-safe input schema
    and no output schema (see the module docstring for why it is dropped)."""
    return tool.model_copy(
        update={"inputSchema": sanitize_schema(tool.inputSchema), "outputSchema": None}
    )


def install_vertex_tool_sanitizer(app) -> None:
    """Re-register the ``tools/list`` handler so advertised schemas are
    Vertex-safe. ``tools/call`` is untouched (still validated against FastMCP's
    original internal schema)."""

    async def _list_tools_sanitized():
        tools = await app.list_tools()
        return [sanitize_tool(t) for t in tools]

    app._mcp_server.list_tools()(_list_tools_sanitized)
    logger.info("Vertex/Gemini schema sanitizer installed for tools/list")
