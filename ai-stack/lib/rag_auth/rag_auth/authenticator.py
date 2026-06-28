"""Authenticators — the single seam that resolves a credential to a :class:`Principal`.

An :class:`Authenticator` answers one question: *given a bearer token, who is this?*
It returns a :class:`Principal` on success or ``None`` to mean "not mine — try the next
mechanism". Returning ``None`` (rather than raising) is what lets
:class:`CompositeAuthenticator` (see :mod:`rag_auth.oidc`) chain OIDC -> static-key -> 401:
an OIDC failure falls through to the static keys, so OIDC fails *closed* while break-glass
keys keep working through an IdP outage.

The :class:`StaticKeyAuthenticator` here is the **break-glass / bootstrap / machine** path:
env-configured keys that always work (CI, air-gapped installs, IdP down). At least one
admin break-glass key should exist and be documented — the single most common self-inflicted
outage is "SSO went down and no one can get in".

Note: cookie-session apps (e.g. chat-ui acting as a BFF) populate ``request.state.principal``
from their own session lookup and reuse :func:`rag_auth.deps.require_scope`; the bearer-token
seam here is for API-style services and for the always-available break-glass path.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Mapping
from typing import Optional, Protocol, runtime_checkable

from .rbac import AUTH_LOCAL, Principal, RoleScopes


@runtime_checkable
class Authenticator(Protocol):
    """Resolve a bearer token to a Principal, or ``None`` if this mechanism can't."""

    @property
    def enabled(self) -> bool:
        ...

    async def authenticate(self, token: Optional[str]) -> Optional[Principal]:
        ...


class StaticKeyAuthenticator:
    """Constant-time match of a bearer token against a fixed set of env-configured keys."""

    def __init__(self, keys: Mapping[str, Principal]):
        self._keys: dict[str, Principal] = dict(keys)

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    async def authenticate(self, token: Optional[str]) -> Optional[Principal]:
        if not token:
            return None
        matched: Optional[Principal] = None
        # Compare against EVERY key with a constant-time compare; never early-exit on
        # token content (avoids leaking which/how-many keys match via timing).
        for known, principal in self._keys.items():
            if hmac.compare_digest(token, known):
                matched = principal
        return matched


def build_static_authenticator(
    roles: RoleScopes,
    *,
    env: Optional[Mapping[str, str]] = None,
    admin_role: str = "admin",
    viewer_role: str = "viewer",
) -> StaticKeyAuthenticator:
    """Build the static/break-glass authenticator from environment variables.

    Recognised env vars (all optional):

    * ``BREAK_GLASS_ADMIN_KEY`` — a bearer token that authenticates as ``admin_role``.
      This is the always-available recovery credential; keep it set in production.
    * ``BREAK_GLASS_VIEWER_KEY`` — a read-only break-glass token (``viewer_role``), if
      that role exists in ``roles``.
    * ``RAG_AUTH_KEYS`` — additional machine identities as a comma-separated list of
      ``name:role:token`` triples, e.g. ``ci:viewer:abc123,ops:admin:def456``.

    A key whose role is unknown to ``roles`` is skipped (keeps boot resilient to typos).
    """
    e = os.environ if env is None else env
    keys: dict[str, Principal] = {}

    def add(token: Optional[str], role: str, name: str) -> None:
        if not token or role not in roles:
            return
        keys[token] = Principal(
            subject=f"key:{name}",
            scopes=roles.scopes_for_role(role),
            auth_method=AUTH_LOCAL,
        )

    add(e.get("BREAK_GLASS_ADMIN_KEY"), admin_role, "break-glass-admin")
    add(e.get("BREAK_GLASS_VIEWER_KEY"), viewer_role, "break-glass-viewer")

    raw = e.get("RAG_AUTH_KEYS", "")
    for entry in (s.strip() for s in raw.split(",") if s.strip()):
        parts = entry.split(":", 2)
        if len(parts) != 3:
            continue
        name, role, token = (p.strip() for p in parts)
        add(token, role, name)

    return StaticKeyAuthenticator(keys)
