"""Tests for build_google_auth wiring (Opción B env-var gating)."""

from mcp_server_odoo import server as srv


def test_no_auth_by_default(monkeypatch):
    monkeypatch.delenv("MCP_REQUIRE_AUTH", raising=False)
    auth, verifier = srv.build_google_auth()
    assert auth is None
    assert verifier is None


def test_auth_enabled_by_env(monkeypatch):
    monkeypatch.setenv("MCP_REQUIRE_AUTH", "1")
    monkeypatch.setenv("MCP_ALLOWED_DOMAIN", "kreilabs.com")
    monkeypatch.setenv("OAUTH_RESOURCE_SERVER_URL", "https://x-uc.a.run.app/mcp")
    auth, verifier = srv.build_google_auth()
    assert auth is not None
    assert verifier is not None
    assert verifier.allowed_domain == "kreilabs.com"


def test_auth_off_when_flag_is_zero(monkeypatch):
    monkeypatch.setenv("MCP_REQUIRE_AUTH", "0")
    auth, verifier = srv.build_google_auth()
    assert auth is None
    assert verifier is None
