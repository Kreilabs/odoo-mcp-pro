"""XML-RPC transports with socket timeouts, plus stale-socket retry policy.

stdlib ``xmlrpc.client.ServerProxy`` has no timeout parameter: an
unresponsive host (firewall dropping packets instead of refusing) holds
the connect for the kernel TCP timeout, ~130s on Linux. Any code path
that probes a user-supplied Odoo URL must use one of these transports.
"""

import http.client
import xmlrpc.client
from urllib.parse import urlparse

# Per-socket-operation timeout for XML-RPC calls to Odoo servers.
DEFAULT_XMLRPC_TIMEOUT = 30

# Errors proving the request never left this process, so replaying it cannot
# duplicate a write.
#
# `CannotSendRequest` is raised by `http.client.putrequest()` when the
# connection state is not IDLE; `ResponseNotReady` by `getresponse()` when the
# state is not REQ_SENT. Both mean the socket was reused in a bad state and
# nothing was transmitted.
PRESEND_ERRORS = (http.client.CannotSendRequest, http.client.ResponseNotReady)

# Odoo model methods that only read. An OSError can surface either while
# sending or while reading the response, so for anything outside this set we
# refuse to replay: a create/write that actually reached Odoo would be applied
# twice. Same reasoning as the "cannot marshal None" note in
# odoo_connection/orm.py — never risk a double post.
READ_ONLY_METHODS = frozenset(
    {
        "search",
        "read",
        "search_read",
        "search_count",
        "fields_get",
        "check_access_rights",
        "name_search",
        "name_get",
        "default_get",
        "read_group",
    }
)


def is_retryable_transport_error(exc: BaseException, method: str) -> bool:
    """Whether ``exc`` is a dead-socket failure worth replaying once.

    Cloud Run freezes instances between requests, which reaps the idle TCP
    socket held by the pooled ``xmlrpc.client.Transport``. The next call then
    fails on a corpse. ``Transport.single_request`` closes the connection on any
    exception, so the retry builds a fresh one — but the stdlib's own retry
    (``Transport.request``) only covers ``RemoteDisconnected`` and ``OSError``
    with errno in ECONNRESET/ECONNABORTED/EPIPE. The failures actually seen in
    production fall outside that set: EBADF (errno 9) is an ``OSError`` with the
    wrong errno, and the two pre-send errors are ``HTTPException``, not
    ``OSError``, so neither ``except`` clause sees them.

    Args:
        exc: The exception raised by the XML-RPC call.
        method: The Odoo method being invoked, used to gate write safety.

    Returns:
        True if the call should be attempted once more.
    """
    if isinstance(exc, PRESEND_ERRORS):
        return True
    # socket.timeout is TimeoutError (an OSError) on 3.10+. A timeout means the
    # request did go out; execute_kw already reports it distinctly, and
    # replaying would double the wait.
    if isinstance(exc, TimeoutError):
        return False
    if isinstance(exc, OSError):
        return method in READ_ONLY_METHODS
    return False


class TimeoutTransport(xmlrpc.client.Transport):
    """HTTP transport that applies a socket timeout."""

    def __init__(self, timeout: float):
        super().__init__()
        self._timeout = timeout

    def make_connection(self, host):
        # The socket does not exist yet at this point; setting
        # ``conn.timeout`` makes http.client pass it to
        # socket.create_connection() and apply it to every recv.
        conn = super().make_connection(host)
        conn.timeout = self._timeout
        return conn


class TimeoutSafeTransport(TimeoutTransport, xmlrpc.client.SafeTransport):
    """HTTPS variant of TimeoutTransport."""


def transport_for_url(url: str, timeout: float) -> TimeoutTransport:
    """Pick the http/https transport matching the URL scheme."""
    if urlparse(url).scheme == "https":
        return TimeoutSafeTransport(timeout)
    return TimeoutTransport(timeout)
