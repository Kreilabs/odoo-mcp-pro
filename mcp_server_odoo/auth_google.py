"""Google-backed TokenVerifier for the single-tenant HTTP transport.

Closes the otherwise-public MCP endpoint: only Bearer tokens that Google's
userinfo endpoint resolves to a user in the allowed Workspace domain are
accepted. Gemini Enterprise sends an opaque Google access token (``ya29.*``),
so validation goes through the OpenID Connect userinfo endpoint (not JWKS).

Enabled from ``server.py`` via env vars; when unset the server stays
unauthenticated (stdio / local usage unchanged).
"""

from __future__ import annotations

import logging
import time

import httpx
from mcp.server.auth.provider import AccessToken, TokenVerifier

logger = logging.getLogger(__name__)

GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


class GoogleTokenVerifier(TokenVerifier):
    """Validate a Google Bearer access token and restrict to one Workspace domain."""

    def __init__(self, allowed_domain: str, cache_ttl: int = 60):
        self.allowed_domain = allowed_domain.lower()
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, AccessToken]] = {}

    async def verify_token(self, token: str) -> AccessToken | None:
        now = time.monotonic()
        hit = self._cache.get(token)
        if hit and hit[0] > now:
            return hit[1]

        try:
            timeout = httpx.Timeout(10.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(
                    GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {token}"},
                )
        except Exception as e:  # network / timeout — fail closed
            logger.warning("Google userinfo call failed: %s", e)
            return None

        if resp.status_code != 200:
            logger.debug("Google userinfo returned %s", resp.status_code)
            return None

        data = resp.json()
        email = (data.get("email") or "").lower()
        hd = (data.get("hd") or "").lower()
        allowed = hd == self.allowed_domain or email.endswith("@" + self.allowed_domain)
        if not allowed:
            logger.warning("Rejecting token: domain not allowed (hd=%r)", hd)
            return None

        access = AccessToken(
            token=token,
            client_id=str(data.get("sub", "google")),
            scopes=[],
            expires_at=int(time.time()) + self.cache_ttl,
        )
        self._cache[token] = (now + self.cache_ttl, access)
        return access
