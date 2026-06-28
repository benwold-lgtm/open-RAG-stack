"""B5 tests: the chat proxy. ai-agent is stubbed (no network) so we exercise chat-ui's own
behaviour — turn persistence, history assembly, auto-title, SSE replay, and error mapping."""

import importlib
import json
import sys

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_UI_DB", str(tmp_path / "chat.db"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("CHAT_STREAM_DELAY", "0")  # no artificial pacing in tests
    monkeypatch.setenv("CHAT_DEFAULT_MODEL", "rag-default")
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    c = TestClient(main.app)
    c.__enter__()
    c.post("/api/auth/register", json={"username": "alice", "password": "password123"})  # admin, logged in
    return main, c, monkeypatch


def _stub_agent(main, monkeypatch, answer="Hello world from RAG", capture=None):
    async def fake(messages, model):
        if capture is not None:
            capture["messages"] = messages
            capture["model"] = model
        return {"choices": [{"message": {"role": "assistant", "content": answer}}],
                "sources": [], "citations": []}
    monkeypatch.setattr(main, "_agent_chat", fake)


def _parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(json.loads(block[len("data: "):]))
    return events


# ── happy path ────────────────────────────────────────────────────────────────
def test_chat_persists_turn_and_streams_answer(app_env):
    main, c, mp = app_env
    _stub_agent(main, mp, answer="Paris is the capital")
    cid = c.post("/api/conversations", json={}).json()["id"]

    r = c.post("/api/chat", json={"conversation_id": cid, "content": "What is the capital of France?"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    text = "".join(e["delta"] for e in events if "delta" in e)
    assert text == "Paris is the capital"
    done = events[-1]
    assert done["done"] is True and done["conversation_id"] == cid

    # both turns persisted, in order
    conv = c.get(f"/api/conversations/{cid}").json()
    assert [(m["role"], m["content"]) for m in conv["messages"]] == [
        ("user", "What is the capital of France?"),
        ("assistant", "Paris is the capital"),
    ]


def test_chat_auto_titles_new_conversation(app_env):
    main, c, mp = app_env
    _stub_agent(main, mp)
    cid = c.post("/api/conversations", json={}).json()["id"]
    assert c.get(f"/api/conversations/{cid}").json()["title"] is None

    c.post("/api/chat", json={"conversation_id": cid, "content": "Tell me about RAG pipelines please"})
    assert c.get(f"/api/conversations/{cid}").json()["title"] == "Tell me about RAG pipelines please"


def test_chat_sends_full_history_to_agent(app_env):
    main, c, mp = app_env
    cap = {}
    _stub_agent(main, mp, answer="first-reply", capture=cap)
    cid = c.post("/api/conversations", json={}).json()["id"]

    c.post("/api/chat", json={"conversation_id": cid, "content": "first"})
    assert cap["messages"] == [{"role": "user", "content": "first"}]
    assert cap["model"] == "rag-default"  # configured default used when client names none

    c.post("/api/chat", json={"conversation_id": cid, "content": "second"})
    assert cap["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first-reply"},
        {"role": "user", "content": "second"},
    ]


def test_chat_honours_explicit_model(app_env):
    main, c, mp = app_env
    cap = {}
    _stub_agent(main, mp, capture=cap)
    cid = c.post("/api/conversations", json={}).json()["id"]
    c.post("/api/chat", json={"conversation_id": cid, "content": "hi", "model": "llama-70b"})
    assert cap["model"] == "llama-70b"


# ── guards + errors ───────────────────────────────────────────────────────────
def test_chat_unknown_conversation_404(app_env):
    main, c, mp = app_env
    _stub_agent(main, mp)
    assert c.post("/api/chat", json={"conversation_id": 999, "content": "hi"}).status_code == 404


def test_chat_empty_message_422(app_env):
    main, c, mp = app_env
    _stub_agent(main, mp)
    cid = c.post("/api/conversations", json={}).json()["id"]
    assert c.post("/api/chat", json={"conversation_id": cid, "content": "   "}).status_code == 422


def test_chat_other_users_conversation_404(app_env):
    main, c, mp = app_env
    _stub_agent(main, mp)
    cid = c.post("/api/conversations", json={}).json()["id"]
    bob = TestClient(main.app)
    bob.post("/api/auth/register", json={"username": "bob", "password": "password123"})  # pending
    bob_id = next(u for u in c.get("/api/admin/users").json() if u["username"] == "bob")["id"]
    c.post(f"/api/admin/users/{bob_id}/approve")
    bob.post("/api/auth/login", json={"username": "bob", "password": "password123"})
    assert bob.post("/api/chat", json={"conversation_id": cid, "content": "peek"}).status_code == 404


def test_chat_agent_down_returns_502_and_no_assistant_message(app_env):
    main, c, mp = app_env

    async def boom(messages, model):
        raise httpx.ConnectError("ai-agent unreachable")
    mp.setattr(main, "_agent_chat", boom)

    cid = c.post("/api/conversations", json={}).json()["id"]
    r = c.post("/api/chat", json={"conversation_id": cid, "content": "hi"})
    assert r.status_code == 502
    # the user's message persisted (they can retry); no assistant reply was saved
    conv = c.get(f"/api/conversations/{cid}").json()
    assert [m["role"] for m in conv["messages"]] == ["user"]


def test_chat_requires_session(app_env):
    main, _, mp = app_env
    _stub_agent(main, mp)
    anon = TestClient(main.app)
    assert anon.post("/api/chat", json={"conversation_id": 1, "content": "hi"}).status_code == 401


# ── models proxy ──────────────────────────────────────────────────────────────
def test_models_proxy_lists_ids(app_env):
    main, c, mp = app_env

    async def fake_models():
        return {"object": "list", "data": [{"id": "rag-default", "object": "model"}]}
    mp.setattr(main, "_agent_models", fake_models)
    assert c.get("/api/models").json() == [{"id": "rag-default"}]
