"""TEMPORARY probe middleware (spike).

Logs the STRUCTURE of the incoming ``Authorization`` header — never the full
token — so we can tell whether Gemini Enterprise sends a JWT id_token
(``jwt_segments=3``) or an opaque access token. Remove after Task 1 of the
Opción-B plan (see plan Task 5).
"""

import logging

logger = logging.getLogger(__name__)


class AuthHeaderProbe:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers") or [])
            raw = headers.get(b"authorization", b"").decode()
            if raw:
                scheme, _, tok = raw.partition(" ")
                segments = tok.count(".") + 1 if tok else 0
                logger.info(
                    "AUTH-PROBE scheme=%s token_len=%d jwt_segments=%d prefix=%s",
                    scheme,
                    len(tok),
                    segments,
                    tok[:6],
                )
            else:
                logger.info(
                    "AUTH-PROBE no-authorization-header path=%s",
                    scope.get("path"),
                )
        await self.app(scope, receive, send)
