import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient

from rag_auth import (
    AuthConfigError,
    RoleScopes,
    build_static_authenticator,
    make_authenticate_request,
    require_scope,
    verify_boot_config,
)
from rag_auth.deps import current_principal

ROLES = RoleScopes(
    {
        "admin": {"things:read", "things:write"},
        "viewer": {"things:read"},
    }
)


def build_app() -> FastAPI:
    authenticator = build_static_authenticator(
        ROLES,
        env={"BREAK_GLASS_ADMIN_KEY": "admin-key", "BREAK_GLASS_VIEWER_KEY": "viewer-key"},
        viewer_role="viewer",
    )
    app = FastAPI()
    protected = APIRouter(dependencies=[Depends(make_authenticate_request(authenticator))])

    @protected.get("/things", dependencies=[Depends(require_scope("things:read"))])
    async def list_things():
        return {"ok": True}

    @protected.post("/things", dependencies=[Depends(require_scope("things:write"))])
    async def create_thing():
        return {"created": True}

    @protected.get("/whoami")
    async def whoami(request: Request):
        p = current_principal(request)
        return {"subject": p.subject, "scopes": sorted(p.scopes)}

    app.include_router(protected)
    return app


@pytest.fixture
def client():
    return TestClient(build_app())


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_missing_token_is_401(client):
    assert client.get("/things").status_code == 401


def test_garbage_token_is_401(client):
    assert client.get("/things", headers=_auth("nonsense")).status_code == 401


def test_admin_can_read_and_write(client):
    assert client.get("/things", headers=_auth("admin-key")).status_code == 200
    assert client.post("/things", headers=_auth("admin-key")).status_code == 200


def test_viewer_can_read_not_write(client):
    assert client.get("/things", headers=_auth("viewer-key")).status_code == 200
    r = client.post("/things", headers=_auth("viewer-key"))
    assert r.status_code == 403
    assert "things:write" in r.json()["detail"]


def test_current_principal_exposes_subject_and_scopes(client):
    r = client.get("/whoami", headers=_auth("viewer-key"))
    assert r.status_code == 200
    body = r.json()
    assert body["subject"] == "key:break-glass-viewer"
    assert body["scopes"] == ["things:read"]


# --- fail-closed boot -------------------------------------------------------

def test_boot_refuses_no_auth_in_production():
    with pytest.raises(AuthConfigError):
        verify_boot_config(production=True, any_auth_enabled=False)


def test_boot_allows_no_auth_in_dev():
    assert verify_boot_config(production=False, any_auth_enabled=False) == []


def test_boot_allows_anonymous_override_with_warning():
    warnings = verify_boot_config(
        production=True, any_auth_enabled=False, allow_anonymous=True
    )
    assert any("anonymous" in w.lower() for w in warnings)


def test_boot_warns_oidc_without_break_glass():
    warnings = verify_boot_config(
        production=True, any_auth_enabled=True, oidc_enabled=True, break_glass_present=False
    )
    assert any("break-glass" in w.lower() for w in warnings)


def test_boot_clean_when_oidc_has_break_glass():
    warnings = verify_boot_config(
        production=True, any_auth_enabled=True, oidc_enabled=True, break_glass_present=True
    )
    assert warnings == []
