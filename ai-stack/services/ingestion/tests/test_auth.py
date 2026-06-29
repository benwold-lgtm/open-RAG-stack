"""Auth-seam tests for ingestion: the shared SERVICE_TOKEN gates every data route, but
/health and the inline page-image route stay open, and production with no token fails
closed at boot. `main` is re-imported per case because the auth wiring runs at module load.

NOTE: importing ingestion's `main` pulls its full runtime stack (crawl4ai, pymupdf,
pytesseract, qdrant-client, ...). Run this from an environment with the service's
requirements installed (the ingestion image, or `pip install -r requirements.txt`)."""
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
    # raise_server_exceptions=False: a route that clears auth then fails in its handler
    # (no DB / qdrant unreachable in this unit env) should surface as a 500 response, not a
    # re-raised exception — these tests assert on the auth layer, not on handler success.
    return TestClient(_load_main(monkeypatch, **env).app, raise_server_exceptions=False)


def test_health_is_open_even_with_auth_active(monkeypatch):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    assert c.get("/health").status_code == 200


def test_page_image_route_stays_open(monkeypatch):
    # Browsers load page images as <img> URLs and cannot send a bearer token, so this
    # route is intentionally unguarded. It must not be rejected with 401 (404 is fine —
    # the doc doesn't exist — the point is auth never blocks it).
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    assert c.get("/documents/nope/pages/1/image").status_code != 401


@pytest.mark.parametrize("method,path", [
    ("get", "/documents"),
    ("get", "/collections"),
    ("post", "/ingest/url"),
    ("post", "/collections"),
    ("delete", "/documents/x"),
])
def test_data_routes_reject_missing_token(monkeypatch, method, path):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    assert getattr(c, method)(path).status_code == 401


def test_data_route_rejects_wrong_token(monkeypatch):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    r = c.get("/documents", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_valid_token_passes_auth(monkeypatch):
    # A valid token must clear the auth layer (the handler itself may then 200 or 500
    # depending on backing services — the point is it is neither 401 nor 403).
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN)
    r = c.get("/collections", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code not in (401, 403)


def test_no_token_leaves_data_plane_open(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/documents").status_code != 401


def test_allow_anonymous_disables_enforcement(monkeypatch):
    c = _client(monkeypatch, SERVICE_TOKEN=TOKEN, ALLOW_ANONYMOUS="true")
    assert c.get("/documents").status_code != 401


def test_production_without_token_refuses_to_boot(monkeypatch):
    with pytest.raises(AuthConfigError):
        _load_main(monkeypatch, ENVIRONMENT="production")


def test_production_anonymous_boots(monkeypatch):
    c = _client(monkeypatch, ENVIRONMENT="production", ALLOW_ANONYMOUS="true")
    assert c.get("/documents").status_code != 401
