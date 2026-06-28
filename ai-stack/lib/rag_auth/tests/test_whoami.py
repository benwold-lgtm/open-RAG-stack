from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from rag_auth import (
    Principal,
    RoleScopes,
    build_static_authenticator,
    make_authenticate_request,
    make_whoami_router,
    whoami_payload,
)

ROLES = RoleScopes({"admin": {"a:read", "a:write"}, "viewer": {"a:read"}})


def test_whoami_payload_shape_and_sorted_scopes():
    p = Principal(subject="oidc:bob", scopes=frozenset({"a:write", "a:read"}), auth_method="oidc")
    assert whoami_payload(p) == {
        "subject": "oidc:bob",
        "scopes": ["a:read", "a:write"],  # sorted
        "auth_method": "oidc",
    }


def _client() -> TestClient:
    authenticator = build_static_authenticator(
        ROLES, env={"BREAK_GLASS_VIEWER_KEY": "ro"}, viewer_role="viewer"
    )
    app = FastAPI()
    app.include_router(make_whoami_router(make_authenticate_request(authenticator)))
    return TestClient(app)


def test_whoami_route_returns_scopes_for_authenticated():
    r = _client().get("/auth/me", headers={"Authorization": "Bearer ro"})
    assert r.status_code == 200
    body = r.json()
    assert body["subject"] == "key:break-glass-viewer"
    assert body["scopes"] == ["a:read"]
    assert body["auth_method"] == "local"


def test_whoami_route_401_unauthenticated():
    assert _client().get("/auth/me").status_code == 401
