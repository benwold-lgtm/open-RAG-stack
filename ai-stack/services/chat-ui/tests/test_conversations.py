"""B4 tests: per-user conversation CRUD + message persistence, and the ownership boundary
(a conversation is only ever visible to its owner; non-owners get 404, not 403)."""

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    def _make(**env):
        monkeypatch.setenv("CHAT_UI_DB", str(tmp_path / "chat.db"))
        monkeypatch.setenv("SESSION_SECRET", "test-secret")
        monkeypatch.setenv("COOKIE_SECURE", "false")
        monkeypatch.setenv("ENVIRONMENT", "development")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        sys.modules.pop("main", None)
        main = importlib.import_module("main")
        c = TestClient(main.app)
        c.__enter__()  # run lifespan (init_db + boot)
        return main, c

    return _make


def _admin_client(make_client, **env):
    """A client logged in as the bootstrap (first-user) admin."""
    main, c = make_client(**env)
    c.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    return main, c


# ── CRUD ─────────────────────────────────────────────────────────────────────
def test_create_list_get(make_client):
    _, c = _admin_client(make_client)
    r = c.post("/api/conversations", json={"title": "First"})
    assert r.status_code == 200 and r.json()["message_count"] == 0
    cid = r.json()["id"]

    lst = c.get("/api/conversations").json()
    assert len(lst) == 1 and lst[0]["id"] == cid

    got = c.get(f"/api/conversations/{cid}").json()
    assert got["title"] == "First" and got["messages"] == []


def test_create_without_title_is_allowed(make_client):
    _, c = _admin_client(make_client)
    r = c.post("/api/conversations", json={})
    assert r.status_code == 200 and r.json()["title"] is None


def test_rename(make_client):
    _, c = _admin_client(make_client)
    cid = c.post("/api/conversations", json={}).json()["id"]
    assert c.patch(f"/api/conversations/{cid}", json={"title": "Renamed"}).status_code == 200
    assert c.get(f"/api/conversations/{cid}").json()["title"] == "Renamed"
    assert c.patch(f"/api/conversations/{cid}", json={"title": "   "}).status_code == 422


def test_delete_cascades_messages(make_client):
    main, c = _admin_client(make_client)
    cid = c.post("/api/conversations", json={}).json()["id"]
    c.post(f"/api/conversations/{cid}/messages", json={"role": "user", "content": "x"})
    assert c.delete(f"/api/conversations/{cid}").status_code == 200
    assert c.get(f"/api/conversations/{cid}").status_code == 404
    with main.connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"] == 0


# ── messages + ordering ───────────────────────────────────────────────────────
def test_messages_persist_in_order(make_client):
    _, c = _admin_client(make_client)
    cid = c.post("/api/conversations", json={"title": "A"}).json()["id"]
    assert c.post(f"/api/conversations/{cid}/messages", json={"role": "user", "content": "hi"}).status_code == 200
    c.post(f"/api/conversations/{cid}/messages", json={"role": "assistant", "content": "hello"})

    conv = c.get(f"/api/conversations/{cid}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert [m["content"] for m in conv["messages"]] == ["hi", "hello"]
    assert conv["message_count"] == 2


def test_adding_a_message_bumps_updated_at(make_client):
    main, c = _admin_client(make_client)
    cid = c.post("/api/conversations", json={}).json()["id"]
    with main.connect() as conn:  # pin updated_at to the epoch so the bump is observable
        conn.execute("UPDATE conversations SET updated_at = 0 WHERE id = ?", (cid,))
    c.post(f"/api/conversations/{cid}/messages", json={"role": "user", "content": "hi"})
    with main.connect() as conn:
        ts = conn.execute("SELECT updated_at FROM conversations WHERE id = ?", (cid,)).fetchone()["updated_at"]
    assert ts > 0


def test_list_orders_by_recent_activity(make_client):
    # updated_at is integer-seconds, so set it explicitly to test the ORDER BY deterministically.
    main, c = _admin_client(make_client)
    a = c.post("/api/conversations", json={"title": "A"}).json()["id"]
    b = c.post("/api/conversations", json={"title": "B"}).json()["id"]
    with main.connect() as conn:
        conn.execute("UPDATE conversations SET updated_at = 2000 WHERE id = ?", (a,))  # A most recent
        conn.execute("UPDATE conversations SET updated_at = 1000 WHERE id = ?", (b,))
    assert [x["id"] for x in c.get("/api/conversations").json()] == [a, b]


def test_bad_message_rejected(make_client):
    _, c = _admin_client(make_client)
    cid = c.post("/api/conversations", json={}).json()["id"]
    assert c.post(f"/api/conversations/{cid}/messages", json={"role": "system", "content": "x"}).status_code == 422
    assert c.post(f"/api/conversations/{cid}/messages", json={"role": "user", "content": "  "}).status_code == 422


# ── ownership + auth boundary ─────────────────────────────────────────────────
def test_ownership_isolation(make_client):
    main, c = _admin_client(make_client, REQUIRE_APPROVAL="false")  # alice = admin
    c2 = TestClient(main.app)
    c2.post("/api/auth/register", json={"username": "bob", "password": "password123"})  # active user

    secret = c.post("/api/conversations", json={"title": "secret"}).json()["id"]
    # bob (even though a valid user) cannot touch alice's conversation in any way
    assert c2.get(f"/api/conversations/{secret}").status_code == 404
    assert c2.patch(f"/api/conversations/{secret}", json={"title": "x"}).status_code == 404
    assert c2.delete(f"/api/conversations/{secret}").status_code == 404
    assert c2.post(f"/api/conversations/{secret}/messages", json={"role": "user", "content": "x"}).status_code == 404
    assert c2.get("/api/conversations").json() == []  # bob's own list is empty


def test_requires_session(make_client):
    main, _ = _admin_client(make_client)
    anon = TestClient(main.app)  # no session cookie
    assert anon.get("/api/conversations").status_code == 401
    assert anon.post("/api/conversations", json={}).status_code == 401
