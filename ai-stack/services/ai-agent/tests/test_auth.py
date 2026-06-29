"""Auth-seam tests for ai-agent: the shared SERVICE_TOKEN gates /v1/* but not /health,
and production with no token fails closed at boot. These re-import `main` per case because
the auth wiring runs at module load from the environment."""
import importlib
import sys

import pytest
from fastapi.testclient import TestClient
from rag_auth.deps import AuthConfigError

TOKEN = "test-service-token"
_AUTH_ENV = ("SERVICE_TOKEN", "ENVIRONMENT", "ALLOW_ANONYMOUS")


def _load_main(monkeypatch, **env):
    """Import a fresh copy of main with exactly the given auth env vars set."""
    for k in _AUTH_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def _client(monkeypatch, **env):
    return TestClient(_load_main(monkeypatch, **env).app)


def test_health_is_open_even_with_auth_active(monkeypatch):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    assert c.get("/health").status_code == 200


def test_models_rejects_missing_token(monkeypatch):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    assert c.get("/v1/models").status_code == 401


def test_models_rejects_wrong_token(monkeypatch):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    assert c.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_models_accepts_valid_token(monkeypatch):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    assert c.get("/v1/models", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_chat_completions_rejects_missing_token(monkeypatch):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    r = c.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 401


def test_no_token_leaves_data_plane_open(monkeypatch):
    # Legacy behavior: with no SERVICE_TOKEN configured, auth is inactive.
    c = _client(monkeypatch)
    assert c.get("/v1/models").status_code == 200


def test_allow_anonymous_disables_enforcement(monkeypatch):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN, ALLOW_ANONYMOUS="true")
    assert c.get("/v1/models").status_code == 200


def test_production_without_token_refuses_to_boot(monkeypatch):
    with pytest.raises(AuthConfigError):
        _load_main(monkeypatch, ENVIRONMENT="production")


def test_production_with_token_boots(monkeypatch):
    c = _client(monkeypatch, ENVIRONMENT="production", SERVICE_TOKEN=TOKEN)
    assert c.get("/v1/models", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_production_anonymous_boots(monkeypatch):
    # Trusted-LAN escape hatch: prod + no token is allowed only with ALLOW_ANONYMOUS.
    c = _client(monkeypatch, ENVIRONMENT="production", ALLOW_ANONYMOUS="true")
    assert c.get("/v1/models").status_code == 200


def test_metrics_open_even_when_auth_active(monkeypatch):
    # /metrics is unauthenticated like /health, even with a token configured.
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    c.get("/health")                       # generate a request sample
    r = c.get("/metrics")                  # no Authorization header
    assert r.status_code == 200
    assert "http_request" in r.text        # Prometheus HTTP metrics are present
