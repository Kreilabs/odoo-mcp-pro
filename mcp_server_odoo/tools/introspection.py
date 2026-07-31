"""Introspection MCP tools: list_models, get_model_fields, list_resource_templates,
server_info."""

from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.types import ToolAnnotations

from ..access_control import AccessControlError
from ..error_handling import ValidationError
from ..error_sanitizer import ErrorSanitizer
from ..logging_config import perf_logger
from ..odoo_connection import OdooConnectionError
from ..schemas import (
    ModelFieldsResult,
    ModelsResult,
    ResourceTemplatesResult,
    ServerInfoResult,
)
from ._common import _current_sub, logger, run_blocking
from ._field_hints import is_writable_field

# Odoo help texts run long; enough to disambiguate a field, not enough to
# crowd out the field list itself.
_MAX_HELP_CHARS = 200


def _to_field_definition(name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Map one entry of a ``fields_get`` response to a FieldDefinition dict."""
    raw_selection = meta.get("selection")
    selection = []
    if isinstance(raw_selection, (list, tuple)):
        for option in raw_selection:
            # Odoo yields [value, label] pairs; skip anything malformed rather
            # than failing the whole introspection call.
            if isinstance(option, (list, tuple)) and len(option) == 2:
                selection.append({"value": str(option[0]), "label": str(option[1])})

    help_text = meta.get("help") or None
    if help_text and len(help_text) > _MAX_HELP_CHARS:
        help_text = help_text[:_MAX_HELP_CHARS].rstrip() + "..."

    return {
        "name": name,
        "type": meta.get("type") or "unknown",
        "string": meta.get("string") or name,
        "required": bool(meta.get("required")),
        "readonly": bool(meta.get("readonly")),
        "relation": meta.get("relation") or None,
        "selection": selection,
        "help": help_text,
    }


class IntrospectionToolsMixin:
    """list_models, get_model_fields, list_resource_templates and server_info tools."""

    def _register_introspection_tools(self):
        """Register introspection tool handlers with FastMCP."""

        @self.app.tool(
            title="List Models",
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        )
        async def list_models(connection: Optional[str] = None) -> ModelsResult:
            """List all models enabled for MCP access with their allowed operations.

            Args:
                connection: Optional. Target a specific Odoo connection by the id
                    from server_info's `connections` list. Hosted multi-tenant
                    only; ignored when self-hosting a single connection.

            Returns:
                List of models with their technical names, display names,
                and allowed operations (read, write, create, unlink).
            """
            result = await self._handle_list_models_tool(connection)
            self._track_usage(_current_sub.get(), "list_models")
            return ModelsResult(**result)

        @self.app.tool(
            title="Get Model Fields",
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        )
        async def get_model_fields(
            model: str,
            include_readonly: bool = False,
            connection: Optional[str] = None,
        ) -> ModelFieldsResult:
            """Get the real field names and types of an Odoo model.

            Call this BEFORE create_record, update_record, create_records or
            import_records on any model you have not already introspected in
            this conversation, and before filtering a search on an unfamiliar
            field.

            Odoo field names are NOT guessable. They are defined per database
            and are often in the database's own language, so a model may use
            `titulo` rather than `name`, or `fecha` rather than `date`. Writing
            a guessed name fails the whole call.

            Args:
                model: The Odoo model name (e.g., 'res.partner')
                include_readonly: Include fields that cannot be written.
                    Default False, which returns the fields you can actually
                    set when creating or updating a record.
                connection: Optional. Target a specific Odoo connection by the id
                    from server_info's `connections` list. Hosted multi-tenant
                    only; ignored when self-hosting a single connection.

            Returns:
                Field definitions with technical name, type, whether the field
                is required, the target model for relations, and the allowed
                values for selection fields.
            """
            result = await self._handle_get_model_fields_tool(model, include_readonly, connection)
            self._track_usage(_current_sub.get(), "get_model_fields")
            return ModelFieldsResult(**result)

        @self.app.tool(
            title="List Resource Templates",
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        )
        async def list_resource_templates() -> ResourceTemplatesResult:
            """List available resource URI templates.

            Since MCP resources with parameters are registered as templates,
            they don't appear in the standard resource list. This tool provides
            information about available resource patterns you can use.

            Returns:
                Resource template definitions with examples and enabled models.
            """
            result = await self._handle_list_resource_templates_tool()
            self._track_usage(_current_sub.get(), "list_resource_templates")
            return ResourceTemplatesResult(**result)

        @self.app.tool(
            title="Server Info",
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        )
        async def server_info() -> ServerInfoResult:
            """Get MCP server version and connection status.

            Returns:
                Server version, git commit, API version, and Odoo connection status.
            """
            from ..server import _BUILD_ORIGIN, GIT_COMMIT, SERVER_VERSION

            error = None
            try:
                connection, _ac, _sub = await self._get_user_context()
                is_connected = (
                    connection.is_authenticated
                    if hasattr(connection, "is_authenticated")
                    else False
                )
                api_version = self.config.api_version if self.config else "json2"
                # Use the connection's actual URL (tenant URL), not the global config.
                # Both XML-RPC and JSON/2 connections expose _base_url; the getattr
                # fallback is just defensive in case of a stale connection object.
                odoo_url = getattr(connection, "_base_url", None) or (
                    self.config.url if self.config else "multi-tenant"
                )
                database = getattr(connection, "database", None)
            except Exception as e:
                # Surface WHY we are not connected so the AI client can tell the
                # user (e.g. "your API key was refused" or "server unreachable")
                # instead of a bare "not connected". Sanitized to keep internals
                # out of the message; the connection resolver already phrases its
                # errors for end users.
                is_connected = False
                api_version = self.config.api_version if self.config else "unknown"
                odoo_url = "not connected"
                database = None
                error = ErrorSanitizer.sanitize_message(str(e))

            # Fetch companies for context (helps with multi-company setups)
            companies = []
            if is_connected:
                try:
                    companies = await run_blocking(
                        connection,
                        connection.search_read,
                        "res.company",
                        [],
                        fields=["id", "name"],
                        limit=10,
                    )
                except Exception:
                    pass

            self._track_usage(_current_sub.get(), "server_info")
            info = ServerInfoResult(
                version=SERVER_VERSION,
                git_commit=GIT_COMMIT,
                api_version=api_version,
                odoo_url=odoo_url,
                database=database,
                connected=is_connected,
                error=error,
                runtime_id=_BUILD_ORIGIN,
                companies=companies,
            )

            # Multi-tenant hook: the hosted handler can list the connections the
            # caller may target (so they can pass a per-call `connection`
            # selector). Standalone returns None, so the key is simply absent.
            available = await self._available_connections()
            if available is not None:
                info.connections = available

            return info

    async def _handle_get_model_fields_tool(
        self,
        model: str,
        include_readonly: bool = False,
        connection_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle get_model_fields tool request."""
        try:
            connection, access_controller, sub = await self._get_user_context(connection_selector)
            with perf_logger.track_operation("tool_get_model_fields", model=model):
                access_controller.validate_model_access(model, "read")

                if not connection.is_authenticated:
                    raise ValidationError("Not authenticated with Odoo")

                # No `attributes` argument: fields_get only consults the
                # 1-hour cache when called for the full definition, and this
                # tool is called repeatedly. Trim the attributes in Python.
                raw = await run_blocking(connection, connection.fields_get, model)

                fields = [
                    _to_field_definition(name, meta)
                    for name, meta in sorted(raw.items())
                    if include_readonly or is_writable_field(meta)
                ]
                # Required first — that is what a failing create is missing.
                fields.sort(key=lambda f: (not f["required"], f["name"]))

                return {"model": model, "fields": fields, "total": len(fields)}

        except ValidationError:
            raise
        except AccessControlError as e:
            raise ValidationError(f"Access denied: {e}") from e
        except OdooConnectionError as e:
            raise ValidationError(f"Connection error: {e}") from e
        except Exception as e:
            logger.error(f"Error in get_model_fields tool: {e}")
            sanitized_msg = ErrorSanitizer.sanitize_message(str(e))
            raise ValidationError(f"Failed to get model fields: {sanitized_msg}") from e

    async def _handle_list_models_tool(
        self, connection_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle list models tool request with permissions."""
        try:
            connection, access_controller, sub = await self._get_user_context(connection_selector)
            with perf_logger.track_operation("tool_list_models"):
                # Get models from MCP access controller
                models = access_controller.get_enabled_models()

                # In JSON/2 mode, get_enabled_models() returns [] because Odoo
                # handles ACLs server-side. Fetch models from ir.model instead.
                if not models and hasattr(connection, "search_read"):
                    try:
                        ir_models = await run_blocking(
                            connection,
                            connection.search_read,
                            "ir.model",
                            [["transient", "=", False]],
                            fields=["model", "name"],
                            order="model asc",
                        )
                        models = [{"model": m["model"], "name": m["name"]} for m in ir_models]
                    except Exception as e:
                        logger.warning(f"Could not fetch models from ir.model: {e}")
                        models = []

                # Return model list without per-model permission checks.
                # Permissions are enforced per-operation to avoid N×4 API calls.
                return {"models": models}
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in list_models tool: {e}")
            sanitized_msg = ErrorSanitizer.sanitize_message(str(e))
            raise ValidationError(f"Failed to list models: {sanitized_msg}") from e

    async def _handle_list_resource_templates_tool(self) -> Dict[str, Any]:
        """Handle list resource templates tool request."""
        try:
            _, access_controller, sub = await self._get_user_context()
            # Get list of enabled models that can be used with resources
            enabled_models = access_controller.get_enabled_models()
            model_names = [m["model"] for m in enabled_models if m.get("read", True)]

            # Define the resource templates
            templates = [
                {
                    "uri_template": "odoo://{model}/record/{record_id}",
                    "description": "Get a specific record by ID",
                    "parameters": {
                        "model": "Odoo model name (e.g., res.partner)",
                        "record_id": "Record ID (e.g., 10)",
                    },
                    "example": "odoo://res.partner/record/10",
                },
                {
                    "uri_template": "odoo://{model}/search",
                    "description": "Basic search returning first 10 records",
                    "parameters": {
                        "model": "Odoo model name",
                    },
                    "example": "odoo://res.partner/search",
                    "note": "Query parameters are not supported. Use search_records tool for advanced queries.",
                },
                {
                    "uri_template": "odoo://{model}/count",
                    "description": "Count all records in a model",
                    "parameters": {
                        "model": "Odoo model name",
                    },
                    "example": "odoo://res.partner/count",
                    "note": "Query parameters are not supported. Use search_records tool for filtered counts.",
                },
                {
                    "uri_template": "odoo://{model}/fields",
                    "description": "Get field definitions for a model",
                    "parameters": {"model": "Odoo model name"},
                    "example": "odoo://res.partner/fields",
                },
            ]

            # Return the resource template information
            return {
                "templates": templates,
                "enabled_models": model_names[:10],  # Show first 10 as examples
                "total_models": len(model_names),
                "note": "Resource URIs do not support query parameters. Use tools (search_records, get_record) for advanced operations with filtering, pagination, and field selection.",
            }

        except Exception as e:
            logger.error(f"Error in list_resource_templates tool: {e}")
            sanitized_msg = ErrorSanitizer.sanitize_message(str(e))
            raise ValidationError(f"Failed to list resource templates: {sanitized_msg}") from e
