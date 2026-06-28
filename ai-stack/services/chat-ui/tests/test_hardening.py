"""B8 hardening tests: security response headers are present on served responses."""

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_UI_DB", str(tmp_path / "chat.db"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("ENVIRONMENT", "development")
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    c = TestClient(main.app)
    c.__enter__()
    return main, c


def test_security_headers_on_spa(client):
    _, c = client
    h = c.get("/").headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-Frame-Options"] == "DENY"
    assert h["Referrer-Policy"] == "same-origin"
    csp = h["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp          # no remote/inline scripts
    assert "frame-ancestors 'none'" in csp     # clickjacking guard


def test_security_headers_on_api(client):
    _, c = client
    # applied to every response, including JSON API and health
    assert c.get("/health").headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in c.get("/api/auth/config").headers
