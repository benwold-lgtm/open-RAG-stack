"""B6 wiring tests: the SPA shell and static assets are served. The SPA's behaviour is
client-side JS (not exercised here); these assert the index injects branding and that the
app bundle + vendored libraries are reachable."""

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
    monkeypatch.setenv("BRAND_NAME", "Acme Docs")
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    c = TestClient(main.app)
    c.__enter__()
    return main, c


def test_index_injects_brand_and_loads_app(client):
    _, c = client
    r = c.get("/")
    assert r.status_code == 200
    assert "Acme Docs" in r.text          # brand injected server-side
    assert "__BRAND__" not in r.text       # placeholder fully replaced
    assert "/static/app.js" in r.text
    assert "/static/vendor/marked.min.js" in r.text
    assert "/static/vendor/purify.min.js" in r.text


def test_static_assets_served(client):
    _, c = client
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/static/styles.css").status_code == 200


def test_vendored_libraries_served_intact(client):
    _, c = client
    marked = c.get("/static/vendor/marked.min.js")
    purify = c.get("/static/vendor/purify.min.js")
    assert marked.status_code == 200 and "marked" in marked.text
    assert purify.status_code == 200 and "DOMPurify" in purify.text
