"""The Prometheus /metrics endpoint is exposed and open (no session required)."""
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
    with TestClient(main.app) as c:        # context manager runs the lifespan
        yield c


def test_metrics_endpoint_is_open(client):
    client.get("/health")                  # generate a request sample
    r = client.get("/metrics")             # no session cookie
    assert r.status_code == 200
    assert "http_request" in r.text
