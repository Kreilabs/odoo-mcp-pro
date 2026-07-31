"""Tests for the stale-socket retry in execute_kw.

Cloud Run freezes instances between requests, reaping the idle socket held by
the pooled XML-RPC Transport. The next call fails on the corpse. These tests
pin both halves of the policy: replay when it is provably safe, refuse when
replaying could duplicate a write.
"""

import errno
import http.client
import socket
from unittest.mock import Mock
from xmlrpc.client import Fault

import pytest

from mcp_server_odoo.config import OdooConfig
from mcp_server_odoo.odoo_connection import OdooConnection, OdooConnectionError
from mcp_server_odoo.xmlrpc_transport import is_retryable_transport_error


def _bad_file_descriptor() -> OSError:
    """The exact error seen in production during check_access_rights."""
    return OSError(errno.EBADF, "Bad file descriptor")


class TestIsRetryableTransportError:
    """Unit tests for the retry predicate."""

    @pytest.mark.parametrize(
        "exc",
        [
            http.client.CannotSendRequest("Request-sent"),
            http.client.ResponseNotReady("Idle"),
        ],
    )
    @pytest.mark.parametrize("method", ["search", "create", "write", "unlink"])
    def test_presend_errors_retry_even_for_writes(self, exc, method):
        """Nothing was transmitted, so replaying cannot duplicate anything."""
        assert is_retryable_transport_error(exc, method) is True

    @pytest.mark.parametrize("method", ["search", "read", "fields_get", "check_access_rights"])
    def test_oserror_retries_on_reads(self, method):
        assert is_retryable_transport_error(_bad_file_descriptor(), method) is True

    @pytest.mark.parametrize("method", ["create", "write", "unlink", "load", "button_validate"])
    def test_oserror_does_not_retry_on_writes(self, method):
        """EBADF can fire after the request reached Odoo — never replay a write."""
        assert is_retryable_transport_error(_bad_file_descriptor(), method) is False

    def test_timeout_never_retries(self):
        """socket.timeout is an OSError on 3.10+, but the request did go out."""
        assert is_retryable_transport_error(socket.timeout(), "search") is False

    def test_fault_never_retries(self):
        assert is_retryable_transport_error(Fault(1, "boom"), "search") is False


class TestExecuteKwRetry:
    """The retry as wired into execute_kw."""

    @pytest.fixture
    def config(self):
        return OdooConfig(url="http://localhost:8069", api_key="test_api_key", database="db")

    @pytest.fixture
    def conn(self, config):
        c = OdooConnection(config)
        c._connected = True
        c._authenticated = True
        c._uid = 2
        c._database = "db"
        c._auth_method = "api_key"
        return c

    def test_presend_error_is_replayed_and_succeeds(self, conn, caplog):
        mock_proxy = Mock()
        mock_proxy.execute_kw.side_effect = [
            http.client.CannotSendRequest("Request-sent"),
            [1, 2, 3],
        ]
        conn._object_proxy = mock_proxy

        result = conn.execute_kw("krl.user.activity", "search", [[]], {})

        assert result == [1, 2, 3]
        assert mock_proxy.execute_kw.call_count == 2
        assert "retrying once" in caplog.text

    def test_presend_error_is_replayed_for_a_write(self, conn):
        """Pre-send is safe regardless of the method."""
        mock_proxy = Mock()
        mock_proxy.execute_kw.side_effect = [http.client.ResponseNotReady("Idle"), 577]
        conn._object_proxy = mock_proxy

        assert conn.execute_kw("krl.user.activity", "create", [{"titulo": "x"}], {}) == 577
        assert mock_proxy.execute_kw.call_count == 2

    def test_bad_file_descriptor_is_replayed_on_a_read(self, conn):
        mock_proxy = Mock()
        mock_proxy.execute_kw.side_effect = [_bad_file_descriptor(), 26]
        conn._object_proxy = mock_proxy

        assert conn.execute_kw("krl.user.activity", "search_count", [[]], {}) == 26
        assert mock_proxy.execute_kw.call_count == 2

    def test_bad_file_descriptor_is_not_replayed_on_a_write(self, conn):
        """The request may have reached Odoo — a replay would double-create."""
        mock_proxy = Mock()
        mock_proxy.execute_kw.side_effect = [_bad_file_descriptor(), 999]
        conn._object_proxy = mock_proxy

        with pytest.raises(OdooConnectionError):
            conn.execute_kw("krl.user.activity", "create", [{"titulo": "x"}], {})

        assert mock_proxy.execute_kw.call_count == 1

    def test_gives_up_after_one_replay(self, conn):
        mock_proxy = Mock()
        mock_proxy.execute_kw.side_effect = http.client.CannotSendRequest("Request-sent")
        conn._object_proxy = mock_proxy

        with pytest.raises(OdooConnectionError):
            conn.execute_kw("res.partner", "search", [[]], {})

        assert mock_proxy.execute_kw.call_count == 2

    def test_timeout_is_not_replayed_and_keeps_its_message(self, conn):
        mock_proxy = Mock()
        mock_proxy.execute_kw.side_effect = socket.timeout()
        conn._object_proxy = mock_proxy

        with pytest.raises(OdooConnectionError, match="timeout"):
            conn.execute_kw("res.partner", "search", [[]], {})

        assert mock_proxy.execute_kw.call_count == 1

    def test_fault_is_not_replayed_and_still_reaches_the_fault_handler(self, conn):
        """An Odoo-level error must not be retried, and 'cannot marshal None'
        must keep its void-success special case."""
        mock_proxy = Mock()
        mock_proxy.execute_kw.side_effect = Fault(1, "cannot marshal None unless allow_none")
        conn._object_proxy = mock_proxy

        assert conn.execute_kw("account.move", "button_draft", [[1]], {}) is None
        assert mock_proxy.execute_kw.call_count == 1

    def test_healthy_call_is_not_retried(self, conn):
        mock_proxy = Mock()
        mock_proxy.execute_kw.return_value = [7]
        conn._object_proxy = mock_proxy

        assert conn.execute_kw("res.partner", "search", [[]], {}) == [7]
        assert mock_proxy.execute_kw.call_count == 1
