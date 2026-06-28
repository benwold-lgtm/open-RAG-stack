"""Exceptions for rag_auth.

Two failure shapes matter here:

* :class:`AuthConfigError` — a *boot-time* misconfiguration (e.g. no auth configured
  in production). The service should refuse to start. Never surfaced to a request.
* :class:`OIDCError` — a *request-time* OIDC validation failure (bad/expired/forged
  token, IdP unreachable). The composite authenticator catches this and falls through
  to the next mechanism (local / break-glass), so OIDC fails *closed* while break-glass
  still works.
"""


class AuthConfigError(Exception):
    """Raised at startup when the auth configuration is unsafe (fail-closed boot)."""


class OIDCError(Exception):
    """Raised when an OIDC token cannot be validated. Caught by the composite authenticator."""


class UnknownRole(KeyError):
    """Raised when a role name has no entry in the role->scopes map."""
