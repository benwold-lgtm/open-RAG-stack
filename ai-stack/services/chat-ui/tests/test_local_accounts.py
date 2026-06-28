"""B2 tests: registration → approval → active, login/logout, break-glass, rate-limit, admin API."""

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    """Factory: import main with a fresh temp DB + given env, return a TestClient."""

    def _make(**env):
        monkeypatch.setenv("CHAT_UI_DB", str(tmp_path / "chat.db"))
        monkeypatch.setenv("SESSION_SECRET", "test-secret")
        monkeypatch.setenv("COOKIE_SECURE", "false")
        monkeypatch.setenv("ENVIRONMENT", "development")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        sys.modules.pop("main", None)
        main = importlib.import_module("main")
        client = TestClient(main.app)
        client.__enter__()  # run lifespan (init_db + boot)
        return main, client

    return _make


@pytest.fixture()
def client(make_client):
    return make_client()


# ── bootstrap + registration ─────────────────────────────────────────────────
def test_first_user_becomes_active_admin_and_is_logged_in(client):
    _, c = client
    r = c.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    assert r.status_code == 200
    assert r.json() == {"status": "active", "role": "admin", "username": "alice"}
    # session cookie set → already authenticated
    me = c.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["role"] == "admin" and "users:manage" in me.json()["scopes"]


def test_second_user_is_pending_and_cannot_login(client):
    _, c = client
    c.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    c.post("/api/auth/logout")
    r = c.post("/api/auth/register", json={"username": "bob", "password": "password123"})
    assert r.status_code == 202 and r.json()["status"] == "pending"
    # pending user is blocked at login
    login = c.post("/api/auth/login", json={"username": "bob", "password": "password123"})
    assert login.status_code == 403


def test_duplicate_username_rejected(client):
    _, c = client
    c.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    r = c.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    assert r.status_code == 409


def test_weak_password_and_bad_username_rejected(client):
    _, c = client
    assert c.post("/api/auth/register", json={"username": "ok", "password": "short"}).status_code == 422
    assert c.post("/api/auth/register", json={"username": "a b", "password": "password123"}).status_code == 422


def test_registration_disabled(make_client):
    _, c = make_client(REGISTRATION_ENABLED="false")
    assert c.post("/api/auth/register", json={"username": "x", "password": "password123"}).status_code == 403


def test_no_approval_mode_activates_immediately(make_client):
    _, c = make_client(REQUIRE_APPROVAL="false")
    c.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    c.post("/api/auth/logout")
    r = c.post("/api/auth/register", json={"username": "bob", "password": "password123"})
    assert r.status_code == 200 and r.json()["status"] == "active"


# ── login / logout ───────────────────────────────────────────────────────────
def test_login_logout_cycle(client):
    _, c = client
    c.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    c.post("/api/auth/logout")
    assert c.get("/api/auth/me").status_code == 401
    bad = c.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
    assert bad.status_code == 401
    ok = c.post("/api/auth/login", json={"username": "alice", "password": "password123"})
    assert ok.status_code == 200
    assert c.get("/api/auth/me").json()["username"] == "alice"
    c.post("/api/auth/logout")
    assert c.get("/api/auth/me").status_code == 401


# ── break-glass ──────────────────────────────────────────────────────────────
def test_break_glass_login_works_with_empty_user_table(make_client):
    main, c = make_client(
        BREAK_GLASS_ADMIN_USER="root", BREAK_GLASS_ADMIN_PASSWORD="rescue-pw-123"
    )
    assert main.count_users() == 0  # nobody registered
    r = c.post("/api/auth/login", json={"username": "root", "password": "rescue-pw-123"})
    assert r.status_code == 200 and r.json()["role"] == "admin"
    me = c.get("/api/auth/me")
    assert me.status_code == 200 and "users:manage" in me.json()["scopes"]


def test_break_glass_wrong_password_rejected(make_client):
    _, c = make_client(BREAK_GLASS_ADMIN_USER="root", BREAK_GLASS_ADMIN_PASSWORD="rescue-pw-123")
    assert c.post("/api/auth/login", json={"username": "root", "password": "nope"}).status_code == 401


# ── rate limiting ────────────────────────────────────────────────────────────
def test_login_rate_limited_after_failures(make_client):
    _, c = make_client(LOGIN_RATE_LIMIT="3")
    c.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    for _ in range(3):
        assert c.post("/api/auth/login", json={"username": "alice", "password": "x"}).status_code == 401
    # 4th attempt (even with correct password) is blocked
    blocked = c.post("/api/auth/login", json={"username": "alice", "password": "password123"})
    assert blocked.status_code == 429


# ── admin user management ────────────────────────────────────────────────────
def _register(c, username, pw="password123"):
    return c.post("/api/auth/register", json={"username": username, "password": pw})


def test_admin_can_approve_and_manage(make_client):
    main, c = make_client()
    _register(c, "admin1")  # first → active admin, logged in
    # second user, pending
    c2 = TestClient(main.app)
    _register(c2, "bob")
    # admin lists users and approves bob
    users = c.get("/api/admin/users").json()
    bob = next(u for u in users if u["username"] == "bob")
    assert bob["status"] == "pending"
    assert c.post(f"/api/admin/users/{bob['id']}/approve").status_code == 200
    # bob can now log in
    assert c2.post("/api/auth/login", json={"username": "bob", "password": "password123"}).status_code == 200


def test_non_admin_cannot_reach_admin_api(make_client):
    main, c = make_client(REQUIRE_APPROVAL="false")
    _register(c, "admin1")  # active admin
    c2 = TestClient(main.app)
    _register(c2, "bob")  # active user (no approval mode)
    c2.post("/api/auth/login", json={"username": "bob", "password": "password123"})
    assert c2.get("/api/admin/users").status_code == 403


def test_disable_user_kills_their_session(make_client):
    main, c = make_client(REQUIRE_APPROVAL="false")
    _register(c, "admin1")
    c2 = TestClient(main.app)
    _register(c2, "bob")
    c2.post("/api/auth/login", json={"username": "bob", "password": "password123"})
    assert c2.get("/api/auth/me").status_code == 200
    bob = next(u for u in c.get("/api/admin/users").json() if u["username"] == "bob")
    assert c.post(f"/api/admin/users/{bob['id']}/disable").status_code == 200
    assert c2.get("/api/auth/me").status_code == 401  # session force-killed


def test_cannot_disable_last_admin(client):
    main, c = client
    _register(c, "admin1")
    me_id = next(u for u in c.get("/api/admin/users").json() if u["username"] == "admin1")["id"]
    assert c.post(f"/api/admin/users/{me_id}/disable").status_code == 409


def test_set_role_promote_and_demote_guard(make_client):
    main, c = make_client(REQUIRE_APPROVAL="false")
    _register(c, "admin1")
    c2 = TestClient(main.app)
    _register(c2, "bob")
    bob_id = next(u for u in c.get("/api/admin/users").json() if u["username"] == "bob")["id"]
    # bad role rejected (admin1 still admin)
    assert c.post(f"/api/admin/users/{bob_id}/role", json={"role": "wizard"}).status_code == 422
    # promote bob → now two admins
    assert c.post(f"/api/admin/users/{bob_id}/role", json={"role": "admin"}).status_code == 200
    # admin1 may now demote self (bob remains an active admin)
    a1_id = next(u for u in c.get("/api/admin/users").json() if u["username"] == "admin1")["id"]
    assert c.post(f"/api/admin/users/{a1_id}/role", json={"role": "user"}).status_code == 200
