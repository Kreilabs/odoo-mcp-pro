"""Tests for GoogleTokenVerifier (Opción B — close the public MCP endpoint)."""

import httpx

from mcp_server_odoo.auth_google import GoogleTokenVerifier


class _MockResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _patch_userinfo(monkeypatch, status_code, payload):
    async def fake_get(self, url, headers=None):
        return _MockResp(status_code, payload)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)


async def test_accepts_token_of_allowed_domain(monkeypatch):
    _patch_userinfo(
        monkeypatch,
        200,
        {"sub": "123", "email": "vinojosa@kreilabs.com", "hd": "kreilabs.com"},
    )
    v = GoogleTokenVerifier(allowed_domain="kreilabs.com")
    result = await v.verify_token("tok-abc")
    assert result is not None
    assert result.token == "tok-abc"
    assert result.client_id == "123"


async def test_accepts_by_email_suffix_when_no_hd(monkeypatch):
    _patch_userinfo(
        monkeypatch,
        200,
        {"sub": "5", "email": "user@kreilabs.com", "hd": ""},
    )
    v = GoogleTokenVerifier(allowed_domain="kreilabs.com")
    assert await v.verify_token("tok-2") is not None


async def test_rejects_other_domain(monkeypatch):
    _patch_userinfo(
        monkeypatch,
        200,
        {"sub": "9", "email": "evil@gmail.com", "hd": ""},
    )
    v = GoogleTokenVerifier(allowed_domain="kreilabs.com")
    assert await v.verify_token("tok-x") is None


async def test_rejects_invalid_token(monkeypatch):
    _patch_userinfo(monkeypatch, 401, {})
    v = GoogleTokenVerifier(allowed_domain="kreilabs.com")
    assert await v.verify_token("bad") is None


async def test_caches_result(monkeypatch):
    calls = {"n": 0}

    async def fake_get(self, url, headers=None):
        calls["n"] += 1
        return _MockResp(200, {"sub": "1", "email": "a@kreilabs.com", "hd": "kreilabs.com"})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    v = GoogleTokenVerifier(allowed_domain="kreilabs.com")
    await v.verify_token("same")
    await v.verify_token("same")
    assert calls["n"] == 1  # second call served from cache
