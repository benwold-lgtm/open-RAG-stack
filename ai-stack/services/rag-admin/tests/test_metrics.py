"""rag-admin exposes /metrics, and it stays open even when the optional Basic Auth
gate is enabled — like /health. Other routes still require the credentials."""
import importlib
import sys

from fastapi.testclient import TestClient


def _load_main(monkeypatch, **env):
    for k in ("ADMIN_USER", "ADMIN_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def test_metrics_open_with_basic_auth_enabled(monkeypatch):
    main = _load_main(monkeypatch, ADMIN_USER="admin", ADMIN_PASSWORD="pw")
    c = TestClient(main.app)
    c.get("/health")                       # generate a request sample
    r = c.get("/metrics")                  # no Basic credentials
    assert r.status_code == 200
    assert "http_request" in r.text


def test_basic_auth_still_gates_other_routes(monkeypatch):
    # Sanity: the exemption is scoped to /health + /metrics, not a hole in the gate.
    main = _load_main(monkeypatch, ADMIN_USER="admin", ADMIN_PASSWORD="pw")
    c = TestClient(main.app)
    assert c.get("/health").status_code == 200
    assert c.get("/metrics").status_code == 200
    assert c.get("/").status_code == 401   # gated route still demands credentials
