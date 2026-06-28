"""B1 skeleton tests: DB + sessions + the rag_auth seam + shell + fail-closed boot.

The DB path and a session secret are set in the environment *before* importing main, so the
module picks up a throwaway SQLite file under tmp.
"""

import importlib
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def chat(tmp_path, monkeypatch):
    """Import (or reimport) main with a fresh temp DB and a known session secret."""
    monkeypatch.setenv("CHAT_UI_DB", str(tmp_path / "chat.db"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("COOKIE_SECURE", "false")  # TestClient isn't over TLS
    monkeypatch.setenv("ENVIRONMENT", "development")
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    with TestClient(main.app) as client:  # context manager runs lifespan (init_db + boot)
        yield main, client


def _seed_user(main, *, role="user", status="active", auth_source="local"):
    now = int(time.time())
    with main.connect() as c:
        cur = c.execute(
            "INSERT INTO users(username,email,auth_source,role,status,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (f"u{now}{role}{status}", "x@example.com", auth_source, role, status, now),
        )
        return cur.lastrowid


def _login_cookie(main, user_id):
    token = main.create_session(user_id)
    return {main.COOKIE_NAME: main.cookie_value(token)}


# ── open endpoints ───────────────────────────────────────────────────────────
def test_health_is_open(chat):
    _, client = chat
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_shell_renders_brand(chat):
    main, client = chat
    r = client.get("/")
    assert r.status_code == 200
    assert main.BRAND_NAME in r.text


# ── the seam: no session → 401, valid session → principal ─────────────────────
def test_auth_me_401_without_session(chat):
    _, client = chat
    assert client.get("/api/auth/me").status_code == 401


def test_protected_route_401_without_session(chat):
    _, client = chat
    assert client.get("/api/conversations").status_code == 401


def test_auth_me_returns_scopes_for_active_user(chat):
    main, client = chat
    uid = _seed_user(main, role="user")
    r = client.get("/api/auth/me", cookies=_login_cookie(main, uid))
    assert r.status_code == 200
    body = r.json()
    assert body["subject"] == f"local:{uid}"
    assert body["auth_method"] == "local"
    assert "chat:use" in body["scopes"]
    assert "users:manage" not in body["scopes"]


def test_admin_session_has_users_manage(chat):
    main, client = chat
    uid = _seed_user(main, role="admin")
    r = client.get("/api/auth/me", cookies=_login_cookie(main, uid))
    assert "users:manage" in r.json()["scopes"]


def test_protected_route_passes_with_session(chat):
    main, client = chat
    uid = _seed_user(main, role="user")
    r = client.get("/api/conversations", cookies=_login_cookie(main, uid))
    assert r.status_code == 200 and r.json() == []


# ── session invalidation ─────────────────────────────────────────────────────
def test_disabled_user_session_is_rejected(chat):
    main, client = chat
    uid = _seed_user(main, status="disabled")
    assert client.get("/api/auth/me", cookies=_login_cookie(main, uid)).status_code == 401


def test_tampered_cookie_is_rejected(chat):
    main, client = chat
    uid = _seed_user(main)
    _login_cookie(main, uid)
    r = client.get("/api/auth/me", cookies={main.COOKIE_NAME: "not-a-valid-signed-token"})
    assert r.status_code == 401


def test_deleted_session_is_rejected(chat):
    main, client = chat
    uid = _seed_user(main)
    token = main.create_session(uid)
    cookies = {main.COOKIE_NAME: main.cookie_value(token)}
    assert client.get("/api/auth/me", cookies=cookies).status_code == 200
    main.delete_session(token)
    assert client.get("/api/auth/me", cookies=cookies).status_code == 401


# ── fail-closed boot ─────────────────────────────────────────────────────────
def test_boot_refuses_no_auth_in_production(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_UI_DB", str(tmp_path / "c.db"))
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SESSION_SECRET", "s")
    monkeypatch.setenv("LOCAL_LOGIN_ENABLED", "false")  # and no OIDC, no break-glass
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    with pytest.raises(main.AuthConfigError):
        main.run_boot_checks()


def test_boot_refuses_missing_session_secret_in_production(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_UI_DB", str(tmp_path / "c.db"))
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.setenv("LOCAL_LOGIN_ENABLED", "true")
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    with pytest.raises(main.AuthConfigError):
        main.run_boot_checks()
