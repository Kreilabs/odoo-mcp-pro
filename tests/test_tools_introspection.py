"""Tests for the get_model_fields tool.

The scenario that motivated it: `krl.user.activity` names its fields in
Spanish, an MCP client guessed `name`/`date`/`duracion`, and every write
failed. A client must be able to discover the real names.

Shared tool fixtures live in tests/helpers/tool_fixtures.py.
"""

from unittest.mock import AsyncMock

import pytest

from mcp_server_odoo.access_control import AccessControlError
from mcp_server_odoo.error_handling import ValidationError
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

# Trimmed from the real prod model.
ACTIVITY_FIELDS = {
    "titulo": {"type": "char", "string": "Título", "required": True, "readonly": False},
    "fecha": {"type": "date", "string": "Fecha", "required": True, "readonly": False},
    "destino": {
        "type": "selection",
        "string": "Destino",
        "required": True,
        "readonly": False,
        "selection": [["borrador", "Borrador"], ["ticket", "Ticket"]],
    },
    "user_id": {
        "type": "many2one",
        "string": "Usuario",
        "required": True,
        "readonly": False,
        "relation": "res.users",
    },
    "descripcion": {
        "type": "text",
        "string": "Descripción",
        "required": False,
        "readonly": False,
        "help": "x" * 400,
    },
    "display_name": {
        "type": "char",
        "string": "Display Name",
        "required": False,
        "readonly": True,
    },
    "create_date": {
        "type": "datetime",
        "string": "Created on",
        "required": False,
        "readonly": True,
    },
}


def _patch_user_context(handler, mock_connection, mock_access_controller):
    ctx = AsyncMock(return_value=(mock_connection, mock_access_controller, "stdio"))
    handler._get_user_context = ctx
    return ctx


class TestGetModelFields:
    """Happy paths and the readonly filter."""

    @pytest.mark.asyncio
    async def test_returns_writable_fields_with_required_first(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        _patch_user_context(handler, mock_connection, mock_access_controller)
        mock_connection.fields_get.return_value = ACTIVITY_FIELDS

        result = await mock_app._tools["get_model_fields"](model="krl.user.activity")

        names = [f.name for f in result.fields]
        assert names[:4] == ["destino", "fecha", "titulo", "user_id"], "required first, then a-z"
        assert "descripcion" in names
        assert result.model == "krl.user.activity"
        assert result.total == len(result.fields)

    @pytest.mark.asyncio
    async def test_excludes_readonly_by_default(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        _patch_user_context(handler, mock_connection, mock_access_controller)
        mock_connection.fields_get.return_value = ACTIVITY_FIELDS

        result = await mock_app._tools["get_model_fields"](model="krl.user.activity")

        names = [f.name for f in result.fields]
        assert "display_name" not in names
        assert "create_date" not in names

    @pytest.mark.asyncio
    async def test_include_readonly_returns_everything(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        _patch_user_context(handler, mock_connection, mock_access_controller)
        mock_connection.fields_get.return_value = ACTIVITY_FIELDS

        result = await mock_app._tools["get_model_fields"](
            model="krl.user.activity", include_readonly=True
        )

        names = [f.name for f in result.fields]
        assert "display_name" in names
        assert result.total == len(ACTIVITY_FIELDS)

    @pytest.mark.asyncio
    async def test_maps_selection_relation_and_truncates_help(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        _patch_user_context(handler, mock_connection, mock_access_controller)
        mock_connection.fields_get.return_value = ACTIVITY_FIELDS

        result = await mock_app._tools["get_model_fields"](model="krl.user.activity")
        by_name = {f.name: f for f in result.fields}

        assert [o.value for o in by_name["destino"].selection] == ["borrador", "ticket"]
        assert by_name["destino"].selection[0].label == "Borrador"
        assert by_name["user_id"].relation == "res.users"
        assert by_name["titulo"].selection == []
        assert by_name["titulo"].relation is None
        assert len(by_name["descripcion"].help) < 400

    @pytest.mark.asyncio
    async def test_uses_the_cacheable_fields_get_call(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        """Passing `attributes` would bypass the 1-hour field cache."""
        _patch_user_context(handler, mock_connection, mock_access_controller)
        mock_connection.fields_get.return_value = ACTIVITY_FIELDS

        await mock_app._tools["get_model_fields"](model="krl.user.activity")

        mock_connection.fields_get.assert_called_once_with("krl.user.activity")

    @pytest.mark.asyncio
    async def test_forwards_connection_selector(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        ctx = _patch_user_context(handler, mock_connection, mock_access_controller)
        mock_connection.fields_get.return_value = ACTIVITY_FIELDS

        await mock_app._tools["get_model_fields"](model="krl.user.activity", connection="7")

        ctx.assert_awaited_once_with("7")


class TestGetModelFieldsErrors:
    """Failure paths."""

    @pytest.mark.asyncio
    async def test_access_denied(self, handler, mock_app, mock_connection, mock_access_controller):
        _patch_user_context(handler, mock_connection, mock_access_controller)
        mock_access_controller.validate_model_access.side_effect = AccessControlError("nope")

        with pytest.raises(ValidationError, match="Access denied"):
            await mock_app._tools["get_model_fields"](model="res.partner")

    @pytest.mark.asyncio
    async def test_not_authenticated(
        self, handler, mock_app, mock_connection, mock_access_controller
    ):
        _patch_user_context(handler, mock_connection, mock_access_controller)
        mock_connection.is_authenticated = False

        with pytest.raises(ValidationError, match="Not authenticated"):
            await mock_app._tools["get_model_fields"](model="res.partner")
