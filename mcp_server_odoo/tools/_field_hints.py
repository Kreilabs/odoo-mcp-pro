"""Turn "unknown field" failures into self-correcting errors.

An LLM client that guesses a field name gets back a dead end today: Odoo says
`Invalid field 'duracion' on model 'krl.user.activity'`, the sanitizer trims it
to `Invalid field 'duracion' in request`, and the model has nothing to go on
except another guess. Field names are not guessable — a model may well name
them in the database's own language (`titulo`, `fecha`, `fuente`).

So when a call fails on an unknown field, we answer with the field names that
*do* exist. The client self-corrects on the first failure instead of looping,
without needing any client-side instruction about the model.

The lookup is free in practice: `fields_get` without `attributes` is served
from the 1-hour cache in `performance.py`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..error_sanitizer import ErrorSanitizer
from ._common import logger, run_blocking

# Phrases that mean "you named a field that does not exist". Covers the raw
# Odoo faults and the shapes ErrorSanitizer rewrites them into.
_UNKNOWN_FIELD_SIGNALS = (
    "invalid field",
    "unknown field",
    "does not exist on this model",
)

# A quoted identifier, or a dotted `model.field` path (the "in leaf" variant).
_QUOTED_NAME = re.compile(r"['\"]([a-zA-Z_][a-zA-Z0-9_.]*)['\"]")
_BARE_DOTTED_NAME = re.compile(r"field\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)")

# Enough names to be useful without flooding the context on a model like
# res.partner (~200 fields).
_MAX_LISTED = 40


def is_writable_field(meta: Dict[str, Any]) -> bool:
    """Whether a client could pass this field to create/write.

    Required-but-readonly fields stay in: they are usually computed-and-stored
    and the caller still needs to know they exist.
    """
    return bool(meta.get("required")) or not meta.get("readonly")


def extract_unknown_field(message: str) -> Optional[str]:
    """Return the offending field name if ``message`` is an unknown-field error.

    Args:
        message: The error text, sanitized or raw.

    Returns:
        The bare field name, or None if this is a different kind of error.
    """
    lowered = message.lower()
    if not any(signal in lowered for signal in _UNKNOWN_FIELD_SIGNALS):
        return None

    match = _QUOTED_NAME.search(message) or _BARE_DOTTED_NAME.search(message)
    if not match:
        return None
    # `krl.user.activity.date` -> `date`; a bare name is unchanged.
    return match.group(1).rsplit(".", maxsplit=1)[-1]


def _summarize(name: str, meta: Dict[str, Any]) -> str:
    ftype = meta.get("type") or "unknown"
    return f"{name} ({ftype}, required)" if meta.get("required") else f"{name} ({ftype})"


async def enrich_unknown_field_error(
    exc: BaseException, connection: Any, model: str
) -> Optional[str]:
    """Build a self-correcting message for an unknown-field failure.

    Args:
        exc: The exception raised by the Odoo call.
        connection: The live connection, used to look up the real fields.
        model: The model the caller was writing to or querying.

    Returns:
        A message naming the valid fields, or None when ``exc`` is not an
        unknown-field error (or the lookup itself fails, in which case the
        caller should fall back to the normal sanitized message).
    """
    if connection is None:
        return None

    unknown = extract_unknown_field(str(exc))
    if unknown is None:
        return None

    try:
        fields: Dict[str, Dict[str, Any]] = await run_blocking(
            connection, connection.fields_get, model
        )
    except Exception as lookup_error:
        # Never let the hint path mask the original failure.
        logger.debug(f"Could not fetch fields for {model} to enrich error: {lookup_error}")
        return None

    # Defensive: a backend that hands back something other than the documented
    # mapping must not blow up the error path and mask the original failure.
    if not isinstance(fields, dict) or not fields:
        return None

    writable = {name: meta for name, meta in fields.items() if is_writable_field(meta)}
    # Required first, then alphabetical — the required ones are what a failing
    # create is usually missing.
    ordered = sorted(writable.items(), key=lambda kv: (not kv[1].get("required"), kv[0]))

    listed = [_summarize(name, meta) for name, meta in ordered[:_MAX_LISTED]]
    overflow = len(ordered) - len(listed)
    if overflow > 0:
        listed.append(f"... ({overflow} more)")

    return (
        f"Field '{unknown}' does not exist on {model}. "
        f"Valid writable fields: {', '.join(listed)}. "
        f"Call get_model_fields('{model}') for types, selection values and relations. "
        f"Do not guess field names — they may be in the database's own language."
    )


async def describe_error(exc: BaseException, connection: Any, model: str) -> str:
    """Error text for a tool handler: field-aware when it can be, sanitized otherwise.

    Args:
        exc: The exception being reported.
        connection: The live connection, or None if the failure happened before
            one was resolved.
        model: The model the tool was operating on.

    Returns:
        A message safe to return to the client.
    """
    return await enrich_unknown_field_error(exc, connection, model) or (
        ErrorSanitizer.sanitize_message(str(exc))
    )
