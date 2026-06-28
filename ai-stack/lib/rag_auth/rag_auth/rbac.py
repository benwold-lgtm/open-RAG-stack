"""The scope/role model — the heart of the framework.

Two load-bearing rules:

1. **Authorize on scopes, never role strings.** A request resolves to a
   :class:`Principal` carrying a set of *scopes*; every route guards on a single scope.
2. **Roles are just named bundles of scopes** held in one place (:class:`RoleScopes`).
   Adding or retiring a role is a one-line data change in the consumer, never a code
   change across routes.

This module is **domain-agnostic**: it does not define any scope names. The consuming
service defines its own scope strings and the role->scopes table, e.g. for chat-ui::

    from rag_auth import RoleScopes

    CHAT_USE      = "chat:use"
    CONVOS_READ   = "convos:read"
    CONVOS_WRITE  = "convos:write"
    MODELS_READ   = "models:read"
    USERS_MANAGE  = "users:manage"

    ROLES = RoleScopes({
        "admin": {CHAT_USE, CONVOS_READ, CONVOS_WRITE, MODELS_READ, USERS_MANAGE},
        "user":  {CHAT_USE, CONVOS_READ, CONVOS_WRITE, MODELS_READ},
    })
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .errors import UnknownRole

# Recognised proofs of identity, for audit. Kept as plain strings (not an enum) so a
# new mechanism behind the authenticate() seam needs no change here.
AUTH_LOCAL = "local"      # a local account (password) or break-glass key
AUTH_OIDC = "oidc"        # a validated OIDC JWT
AUTH_NONE = "none"        # unauthenticated (anonymous)


@dataclass(frozen=True)
class Principal:
    """Who is making a request and what they may do.

    ``scopes`` is the *only* thing routes consult. ``auth_method`` is for audit and is
    never used for authorization decisions.
    """

    subject: str
    scopes: frozenset[str]
    auth_method: str

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    def has_any(self, scopes: Iterable[str]) -> bool:
        return any(s in self.scopes for s in scopes)


# A principal with zero scopes. Authenticated-but-unmapped OIDC users and the
# local-dev open mode both resolve to *a* principal with no scopes rather than a 401,
# so the audit log records *who* was denied. Every scoped route 403s for it.
ANONYMOUS = Principal(subject="anonymous", scopes=frozenset(), auth_method=AUTH_NONE)


class RoleScopes:
    """An immutable role -> scopes table.

    The consumer constructs one of these from its own role map. Many IdP groups can map
    to many roles; :meth:`scopes_for_roles` returns the *union* of their scopes, so
    overlapping group membership simply adds capability with no special cases.
    """

    def __init__(self, mapping: Mapping[str, Iterable[str]]):
        self._map: dict[str, frozenset[str]] = {
            role: frozenset(scopes) for role, scopes in mapping.items()
        }

    def scopes_for_role(self, role: str) -> frozenset[str]:
        try:
            return self._map[role]
        except KeyError as exc:
            raise UnknownRole(role) from exc

    def scopes_for_roles(self, roles: Iterable[str]) -> frozenset[str]:
        """Union of scopes for the given roles. Unknown role names are ignored
        (an IdP can assert groups this app has never heard of)."""
        out: set[str] = set()
        for role in roles:
            out |= self._map.get(role, frozenset())
        return frozenset(out)

    def __contains__(self, role: str) -> bool:
        return role in self._map

    @property
    def roles(self) -> frozenset[str]:
        return frozenset(self._map)

    @property
    def all_scopes(self) -> frozenset[str]:
        if not self._map:
            return frozenset()
        return frozenset().union(*self._map.values())
