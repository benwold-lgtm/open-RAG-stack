"""B3 tests: the OIDC Relying-Party flow — login redirect (PKCE/state/nonce), callback
(code exchange, ID-token validation, nonce/state guards), user auto-provisioning, and the
/api/auth/config surface. Fully offline: a local RSA keypair signs ID tokens and a seeded
JWKS verifies them; the IdP's token endpoint is an httpx MockTransport.
"""

import importlib
import json
import sys
import time
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

ISSUER = "https://idp.test/realms/corp"
CLIENT_ID = "chat-ui"
KID = "test-kid-1"


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks(rsa_key):
    pub = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(rsa_key.public_key()))
    pub.update({"kid": KID, "use": "sig", "alg": "RS256"})
    return [pub]


@pytest.fixture()
def oidc_env(tmp_path, monkeypatch):
    """Import main with OIDC enabled and a fresh DB. Returns the main module + TestClient."""
    monkeypatch.setenv("CHAT_UI_DB", str(tmp_path / "chat.db"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("OIDC_ENABLED", "true")
    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("OIDC_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "shh")
    monkeypatch.setenv("OIDC_GROUP_ROLES", "rag-admins:admin,rag-users:user")
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    client = TestClient(main.app)
    client.__enter__()  # run lifespan (init_db + boot)
    return main, client


def _make_provider(main, jwks, rsa_key, token_holder):
    """Build an OIDCProvider wired to a seeded JWKS + a mock token endpoint."""

    async def fetch():
        return jwks

    cfg = main.OIDCConfig(
        issuer=ISSUER,
        audience=CLIENT_ID,
        groups_claim="groups",
        group_roles={"rag-admins": "admin", "rag-users": "user"},
        default_role="user",
    )
    validator = main.OIDCValidator(cfg, main.ROLES, main.JWKSCache(fetch))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"id_token": token_holder["id_token"], "token_type": "Bearer"})
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return main.OIDCProvider(
        client_id=CLIENT_ID,
        client_secret="shh",
        scopes="openid profile email groups",
        authorization_endpoint="https://idp.test/authorize",
        token_endpoint="https://idp.test/token",
        validator=validator,
        username_claim="preferred_username",
        email_claim="email",
        groups_claim="groups",
        group_roles={"rag-admins": "admin", "rag-users": "user"},
        default_role="user",
        http_client=mock_client,
    )


def _id_token(rsa_key, *, nonce, groups, sub="okta|abc", username="alice.sso", email="alice@corp.test"):
    now = int(time.time())
    claims = {
        "sub": sub,
        "iss": ISSUER,
        "aud": CLIENT_ID,
        "iat": now,
        "exp": now + 300,
        "nonce": nonce,
        "preferred_username": username,
        "email": email,
        "groups": groups,
    }
    return jwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": KID})


def _drive_login(main, client, jwks, rsa_key, *, groups, **id_kwargs):
    """Run /oidc/login, read the issued state+nonce, then post back to /oidc/callback.
    Returns the callback response (redirects not followed)."""
    holder = {"id_token": None}
    main._oidc_provider = _make_provider(main, jwks, rsa_key, holder)

    login = client.get("/api/auth/oidc/login", follow_redirects=False)
    assert login.status_code == 302
    q = parse_qs(urlparse(login.headers["location"]).query)
    state, nonce = q["state"][0], q["nonce"][0]
    # PKCE + redirect params present
    assert q["code_challenge_method"][0] == "S256" and q["code_challenge"][0]

    holder["id_token"] = _id_token(rsa_key, nonce=nonce, groups=groups, **id_kwargs)
    return client.get(
        f"/api/auth/oidc/callback?code=auth-code&state={state}", follow_redirects=False
    ), state, nonce


# ── happy path ───────────────────────────────────────────────────────────────
def test_oidc_login_provisions_user_and_sets_session(oidc_env, jwks, rsa_key):
    main, c = oidc_env
    cb, _, _ = _drive_login(main, c, jwks, rsa_key, groups=["rag-users"])
    assert cb.status_code == 302 and cb.headers["location"] == "/"

    me = c.get("/api/auth/me")
    assert me.status_code == 200
    body = me.json()
    assert body["role"] == "user" and body["username"] == "alice.sso"
    assert "chat:use" in body["scopes"] and "users:manage" not in body["scopes"]
    assert main.count_users() == 1


def test_oidc_admin_group_maps_to_admin_role(oidc_env, jwks, rsa_key):
    main, c = oidc_env
    cb, _, _ = _drive_login(main, c, jwks, rsa_key, groups=["rag-admins"], sub="okta|boss", username="boss")
    assert cb.status_code == 302
    assert "users:manage" in c.get("/api/auth/me").json()["scopes"]


def test_repeat_login_same_subject_does_not_duplicate(oidc_env, jwks, rsa_key):
    main, c = oidc_env
    _drive_login(main, c, jwks, rsa_key, groups=["rag-users"])
    c.post("/api/auth/logout")
    _drive_login(main, c, jwks, rsa_key, groups=["rag-users"])
    assert main.count_users() == 1


# ── guards ───────────────────────────────────────────────────────────────────
def test_state_mismatch_rejected(oidc_env, jwks, rsa_key):
    main, c = oidc_env
    holder = {"id_token": None}
    main._oidc_provider = _make_provider(main, jwks, rsa_key, holder)
    login = c.get("/api/auth/oidc/login", follow_redirects=False)
    nonce = parse_qs(urlparse(login.headers["location"]).query)["nonce"][0]
    holder["id_token"] = _id_token(rsa_key, nonce=nonce, groups=["rag-users"])
    # wrong state value (CSRF / mixed-up transaction)
    r = c.get("/api/auth/oidc/callback?code=x&state=forged", follow_redirects=False)
    assert r.status_code == 400


def test_nonce_mismatch_rejected(oidc_env, jwks, rsa_key):
    main, c = oidc_env
    holder = {"id_token": None}
    main._oidc_provider = _make_provider(main, jwks, rsa_key, holder)
    login = c.get("/api/auth/oidc/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    holder["id_token"] = _id_token(rsa_key, nonce="not-the-issued-nonce", groups=["rag-users"])
    r = c.get(f"/api/auth/oidc/callback?code=x&state={state}", follow_redirects=False)
    assert r.status_code == 401  # ID-token validation (nonce) failed


def test_callback_without_transaction_cookie_rejected(oidc_env, jwks, rsa_key):
    main, c = oidc_env
    main._oidc_provider = _make_provider(main, jwks, rsa_key, {"id_token": None})
    # no prior /login → no oidc_tx cookie
    r = c.get("/api/auth/oidc/callback?code=x&state=y", follow_redirects=False)
    assert r.status_code == 400


def test_idp_error_param_surfaced(oidc_env, jwks, rsa_key):
    main, c = oidc_env
    main._oidc_provider = _make_provider(main, jwks, rsa_key, {"id_token": None})
    r = c.get("/api/auth/oidc/callback?error=access_denied", follow_redirects=False)
    assert r.status_code == 400


# ── config surface + disabled behaviour ──────────────────────────────────────
def test_auth_config_reports_oidc_enabled(oidc_env):
    _, c = oidc_env
    cfg = c.get("/api/auth/config").json()
    assert cfg["oidc"]["enabled"] is True
    assert cfg["oidc"]["login_path"] == "/api/auth/oidc/login"
    assert cfg["local_login"] is True


def test_oidc_login_404_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_UI_DB", str(tmp_path / "chat.db"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("OIDC_ENABLED", raising=False)
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    with TestClient(main.app) as c:
        assert c.get("/api/auth/oidc/login", follow_redirects=False).status_code == 404
        assert c.get("/api/auth/config").json()["oidc"]["enabled"] is False


def test_local_login_still_works_when_oidc_enabled(oidc_env):
    main, c = oidc_env
    c.post("/api/auth/register", json={"username": "alice", "password": "password123"})
    c.post("/api/auth/logout")
    ok = c.post("/api/auth/login", json={"username": "alice", "password": "password123"})
    assert ok.status_code == 200
