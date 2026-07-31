"""Tests for self-correcting unknown-field errors.

The production loop this breaks: an MCP client guesses `duracion` on
`krl.user.activity`, Odoo rejects it, the sanitized error says only
"Invalid field 'duracion' in request", and the client guesses again. The
error must name the fields that actually exist.

Shared tool fixtures live in tests/helpers/tool_fixtures.py.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_server_odoo.error_handling import ValidationError
from mcp_server_odoo.exceptions import OdooConnectionError
from mcp_server_odoo.tools._field_hints import (
    describe_error,
    enrich_unknown_field_error,
    extract_unknown_field,
    is_writable_field,
)
from tests.helpers.tool_fixtures import (
    handler,
    mock_access_controller,
    mock_app,
    mock_connection,
    valid_config,
)

__all__ = [
    "handler",
    "mock_access_controller",
    "mock_app",
    "mock_connection",
    "valid_config",
]

ACTIVITY_FIELDS = {
    "titulo": {"type": "char", "string": "Título", "required": True, "readonly": False},
    "fecha": {"type": "date", "string": "Fecha", "required": True, "readonly": False},
    "descripcion": {"type": "text", "string": "Descripción", "readonly": False},
    "display_name": {"type": "char", "string": "Display Name", "readonly": True},
}


class TestExtractUnknownField:
    """Detecting the error and pulling the offending name out of it."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            # As sanitized by ErrorSanitizer, which is what the tool layer sees.
            ("Operation failed: Invalid field 'duracion' in request", "duracion"),
            ("Invalid field 'invalid_field' in search criteria", "invalid_field"),
            ("Field 'bogus' does not exist on this model", "bogus"),
            # Raw Odoo faults, in case they ever reach us unsanitized.
            ("Invalid field 'duracion' on model 'krl.user.activity'", "duracion"),
            ("Invalid field krl.user.activity.date in leaf ('date', '>=', '2026-07-27')", "date"),
        ],
    )
    def test_recognizes_unknown_field_errors(self, message, expected):
        assert extract_unknown_field(message) == expected

    @pytest.mark.parametrize(
        "message",
        [
            "Operation failed: Access denied",
            "Operation timeout after 30 seconds",
            "Record ID 42 not found",
            "",
        ],
    )
    def test_ignores_other_errors(self, message):
        assert extract_unknown_field(message) is None


class TestIsWritableField:
    def test_plain_writable(self):
        assert is_writable_field({"readonly": False}) is True

    def test_readonly_excluded(self):
        assert is_writable_field({"readonly": True}) is False

    def test_required_readonly_kept(self):
        """Usually computed-and-stored; the caller still needs to know it exists."""
        assert is_writable_field({"readonly": True, "required": True}) is True


class TestEnrichUnknownFieldError:
    """Building the corrective message."""

    @pytest.mark.asyncio
    async def test_lists_the_real_fields(self, mock_connection):
        mock_connection.fields_get.return_value = ACTIVITY_FIELDS

        msg = await enrich_unknown_field_error(
            OdooConnectionError("Operation failed: Invalid field 'duracion' in request"),
            mock_connection,
            "krl.user.activity",
        )

        assert "'duracion' does not exist on krl.user.activity" in msg
        assert "titulo (char, required)" in msg
        assert "fecha (date, required)" in msg
        assert "get_model_fields('krl.user.activity')" in msg
        # Readonly fields are noise when you are trying to write.
        assert "display_name" not in msg

    @pytest.mark.asyncio
    async def test_returns_none_for_unrelated_errors(self, mock_connection):
        msg = await enrich_unknown_field_error(
            OdooConnectionError("Operation timeout after 30 seconds"),
            mock_connection,
            "krl.user.activity",
        )
        assert msg is None
        mock_connection.fields_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_without_a_connection(self):
        msg = await enrich_unknown_field_error(
            OdooConnectionError("Invalid field 'x' in request"), None, "res.partner"
        )
        assert msg is None

    @pytest.mark.asyncio
    async def test_lookup_failure_never_masks_the_original_error(self, mock_connection):
        mock_connection.fields_get.side_effect = RuntimeError("odoo is down")

        msg = await enrich_unknown_field_error(
            OdooConnectionError("Invalid field 'x' in request"), mock_connection, "res.partner"
        )
        assert msg is None

    @pytest.mark.asyncio
    async def test_non_dict_response_is_survivable(self, mock_connection):
        """A MagicMock (or any non-mapping) must not blow up the error path."""
        mock_connection.fields_get.return_value = MagicMock()

        msg = await enrich_unknown_field_error(
            OdooConnectionError("Invalid field 'x' in request"), mock_connection, "res.partner"
        )
        assert msg is None

    @pytest.mark.asyncio
    async def test_caps_the_list_on_a_large_model(self, mock_connection):
        mock_connection.fields_get.return_value = {
            f"field_{i:03d}": {"type": "char", "readonly": False} for i in range(200)
        }

        msg = await enrich_unknown_field_error(
            OdooConnectionError("Invalid field 'nope' in request"), mock_connection, "res.partner"
        )

        assert "more)" in msg
        assert msg.count("(char)") <= 40

    @pytest.mark.asyncio
    async def test_describe_error_falls_back_to_the_sanitizer(self, mock_connection):
        text = await describe_error(
            OdooConnectionError("Operation timeout after 30 seconds"),
            mock_connection,
            "res.partner",
        )
        assert "timed out" in text.lower() or "timeout" in text.lower()


class TestEndToEndThroughTheTools:
    """The prod failure, reproduced through the real tool handlers."""

    @staticmethod
    def _patch(handler, mock_connection, mock_access_controller):
        handler._get_user_context = AsyncMock(
            return_value=(mock_connection, mock_access_controller, "stdio")
        )

    @pytest.mark.asyncio
    async def test_create_record_with_a_guessed_field_names_the_real_ones(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        self._patch(handler, mock_connection, mock_access_controller)
        mock_connection.create.side_effect = OdooConnectionError(
            "Operation failed: Invalid field 'duracion' in request"
        )
        mock_connection.fields_get.return_value = ACTIVITY_FIELDS

        with pytest.raises(ValidationError) as excinfo:
            await mock_app._tools["create_record"](
                model="krl.user.activity",
                values={"name": "Reunión", "duracion": 1.5},
            )

        message = str(excinfo.value)
        assert "titulo" in message
        assert "fecha" in message
        assert "get_model_fields" in message

    @pytest.mark.asyncio
    async def test_search_with_a_guessed_field_names_the_real_ones(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        self._patch(handler, mock_connection, mock_access_controller)
        mock_connection.search_count.side_effect = OdooConnectionError(
            "Operation failed: Invalid field 'date' in search criteria"
        )
        mock_connection.fields_get.return_value = ACTIVITY_FIELDS

        with pytest.raises(ValidationError) as excinfo:
            await mock_app._tools["search_records"](
                model="krl.user.activity", domain='[["date", ">=", "2026-07-27"]]'
            )

        assert "fecha" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_real_connection_error_keeps_its_message(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        self._patch(handler, mock_connection, mock_access_controller)
        mock_connection.create.side_effect = OdooConnectionError("Cannot connect to Odoo server")

        with pytest.raises(ValidationError, match="Connection error"):
            await mock_app._tools["create_record"](
                model="krl.user.activity", values={"titulo": "x"}
            )
