"""The ``/auth/me`` whoami contract — the UI's single source of truth for permissions.

The UI never decides permissions itself; it asks the service "who am I and what may I do?"
and gates every affordance on the returned ``scopes``. Because the service computes those
scopes (from local role or OIDC group mapping), the UI and the API can never drift.

:func:`whoami_payload` is the canonical serialization — both API-style services (which mount
:func:`make_whoami_router`) and cookie-session apps (which build the Principal from their own
session and return this same dict) use it, so the wire shape is identical everywhere.
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, Request

from .deps import current_principal
from .rbac import Principal


def whoami_payload(principal: Principal) -> dict:
    """Serialize a Principal to the ``/auth/me`` contract: ``{subject, scopes, auth_method}``.
    ``scopes`` is sorted for a stable response."""
    return {
        "subject": principal.subject,
        "scopes": sorted(principal.scopes),
        "auth_method": principal.auth_method,
    }


def make_whoami_router(
    authenticate_request: Callable[..., object],
    *,
    path: str = "/auth/me",
) -> APIRouter:
    """Convenience router exposing ``GET {path}`` for API-style services. Authenticated but
    intentionally **not** scope-gated — any authenticated principal may ask who it is."""
    router = APIRouter()

    @router.get(path)
    async def whoami(request: Request, _: None = Depends(authenticate_request)) -> dict:
        return whoami_payload(current_principal(request))

    return router
