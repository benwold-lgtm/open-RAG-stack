"""FastAPI wiring: router-level authentication + per-route scope authorization.

Usage in an API-style service::

    from fastapi import APIRouter, Depends
    from rag_auth import RoleScopes
    from rag_auth.authenticator import build_static_authenticator
    from rag_auth.deps import make_authenticate_request, require_scope

    roles = RoleScopes({...})
    authenticator = build_static_authenticator(roles)        # or a CompositeAuthenticator
    authn = make_authenticate_request(authenticator)

    protected = APIRouter(dependencies=[Depends(authn)])     # authn for every route
    @protected.get("/things", dependencies=[Depends(require_scope("things:read"))])
    async def list_things(): ...

Cookie-session apps set ``request.state.principal`` in their own middleware and reuse
:func:`require_scope` / :func:`current_principal` directly.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .authenticator import Authenticator
from .errors import AuthConfigError
from .rbac import Principal

_bearer = HTTPBearer(auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"}
)


def make_authenticate_request(
    authenticator: Authenticator,
) -> Callable[..., Awaitable[None]]:
    """Build a router-level dependency that resolves the Principal once per request and
    stashes it on ``request.state.principal`` (401 if no mechanism authenticates)."""

    async def authenticate_request(
        request: Request,
        creds: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
    ) -> None:
        token = creds.credentials if creds else None
        principal = await authenticator.authenticate(token)
        if principal is None:
            raise _UNAUTHENTICATED
        request.state.principal = principal

    return authenticate_request


def require_scope(scope: str) -> Callable[..., Awaitable[None]]:
    """Route-level dependency factory: 403 unless the resolved Principal holds ``scope``."""

    async def _dep(request: Request) -> None:
        principal: Optional[Principal] = getattr(request.state, "principal", None)
        if principal is None:
            raise _UNAUTHENTICATED
        if not principal.has(scope):
            raise HTTPException(status_code=403, detail=f"Missing required scope: {scope}")

    return _dep


def current_principal(request: Request) -> Principal:
    """Dependency that returns the resolved Principal (401 if none). For handlers that
    need the subject/scopes (e.g. per-user data isolation)."""
    principal: Optional[Principal] = getattr(request.state, "principal", None)
    if principal is None:
        raise _UNAUTHENTICATED
    return principal


def verify_boot_config(
    *,
    production: bool,
    any_auth_enabled: bool,
    oidc_enabled: bool = False,
    break_glass_present: bool = False,
    allow_anonymous: bool = False,
) -> list[str]:
    """Fail-closed startup check. Call once at service startup.

    * In ``production`` with **no** auth enabled and ``allow_anonymous`` not set, raise
      :class:`AuthConfigError` — otherwise every request would be anonymous-with-full-access.
    * Returns a list of non-fatal warnings the caller should log. The important one:
      OIDC enabled with no break-glass key means an IdP outage locks everyone out.
    """
    if production and not any_auth_enabled and not allow_anonymous:
        raise AuthConfigError(
            "No authentication configured in production (no local accounts, OIDC, or static "
            "keys). Configure a credential, or set allow_anonymous=True only for a trusted "
            "local network."
        )

    warnings: list[str] = []
    if oidc_enabled and not break_glass_present:
        warnings.append(
            "OIDC is enabled but no break-glass admin key is set (BREAK_GLASS_ADMIN_KEY). "
            "An IdP outage would lock everyone out — set a break-glass key."
        )
    if allow_anonymous and production:
        warnings.append(
            "allow_anonymous is set in production — every request runs with no identity. "
            "Use only on a trusted, network-isolated deployment."
        )
    return warnings
