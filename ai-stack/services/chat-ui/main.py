"""chat-ui — first-party multi-user chat service.

Built so far:
* B1 — FastAPI + SQLite + signed server-side sessions, the rag_auth scope seam, a branded
  SPA shell, an open /health, and a fail-closed boot check.
* B2 — local accounts: registration (→ pending → admin-approval → active; first user
  bootstraps as admin), argon2 password hashing, login/logout, a break-glass admin login,
  login rate-limiting, and the admin user-management API.

OIDC (B3), conversations (B4), chat streaming (B5) and the full SPA (B6) build on top.

The session model: a login creates a row in `sessions` and sets a signed, httpOnly, Secure,
SameSite cookie carrying the session token. Every request resolves that cookie back to a
Principal (via rag_auth) so routes guard on scopes, never role strings.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx
from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from rag_auth import (
    AUTH_LOCAL,
    AUTH_OIDC,
    AuthConfigError,
    Principal,
    RoleScopes,
    current_principal,
    require_scope,
    verify_boot_config,
    whoami_payload,
)
from rag_auth.errors import OIDCError
from rag_auth.oidc import JWKSCache, OIDCConfig, OIDCValidator

# ── Config (env) ────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("CHAT_UI_DB", "/data/chat_ui.db")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
SESSION_TTL = int(os.environ.get("SESSION_TTL", str(7 * 24 * 3600)))  # 7 days
COOKIE_NAME = "chat_session"
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() in ("1", "true", "yes")
PRODUCTION = os.environ.get("ENVIRONMENT", "production").lower() == "production"
LOCAL_LOGIN_ENABLED = os.environ.get("LOCAL_LOGIN_ENABLED", "true").lower() in ("1", "true", "yes")
AI_AGENT_URL = os.environ.get("AI_AGENT_URL", "http://ai-agent:8000/v1")
# Shared machine-to-machine token sent to ai-agent (empty when the data plane is open).
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "")
_SERVICE_AUTH = {"Authorization": f"Bearer {SERVICE_TOKEN}"} if SERVICE_TOKEN else {}
# Default model for /api/chat when the client doesn't name one; empty = discover from ai-agent.
DEFAULT_MODEL = os.environ.get("CHAT_DEFAULT_MODEL", "")
AGENT_TIMEOUT = float(os.environ.get("AI_AGENT_TIMEOUT", "120"))  # LLM answers can be slow
STREAM_DELAY = float(os.environ.get("CHAT_STREAM_DELAY", "0.01"))  # word-replay pacing (UX)
BRAND_NAME = os.environ.get("BRAND_NAME", "Open RAG Chat")
BRAND_PRIMARY_COLOR = os.environ.get("BRAND_PRIMARY_COLOR", "#2563eb")

# Local-account policy
REGISTRATION_ENABLED = os.environ.get("REGISTRATION_ENABLED", "true").lower() in ("1", "true", "yes")
REQUIRE_APPROVAL = os.environ.get("REQUIRE_APPROVAL", "true").lower() in ("1", "true", "yes")
MIN_PASSWORD_LEN = int(os.environ.get("MIN_PASSWORD_LEN", "8"))

# Break-glass admin login — a username/password pair, independent of the user table and of
# OIDC, that always logs in as an active admin (recovers an empty/locked-out/IdP-down system).
BREAK_GLASS_USER = os.environ.get("BREAK_GLASS_ADMIN_USER", "")
BREAK_GLASS_PASSWORD = os.environ.get("BREAK_GLASS_ADMIN_PASSWORD", "")
BREAK_GLASS_ENABLED = bool(BREAK_GLASS_USER and BREAK_GLASS_PASSWORD)

# Login rate-limiting (per username+IP). In-memory, single-process — adequate for this scope.
LOGIN_RATE_LIMIT = int(os.environ.get("LOGIN_RATE_LIMIT", "5"))
LOGIN_RATE_WINDOW = int(os.environ.get("LOGIN_RATE_WINDOW", "60"))

# OIDC single-sign-on (this service is the Relying Party). chat-ui runs the login flow
# (PKCE + state + nonce) and validates the IdP's ID token (audience = client_id) via rag_auth.
OIDC_ENABLED = os.environ.get("OIDC_ENABLED", "false").lower() in ("1", "true", "yes")
OIDC_ISSUER = os.environ.get("OIDC_ISSUER", "")
OIDC_CLIENT_ID = os.environ.get("OIDC_CLIENT_ID", "")
OIDC_CLIENT_SECRET = os.environ.get("OIDC_CLIENT_SECRET", "")
OIDC_SCOPES = os.environ.get("OIDC_SCOPES", "openid profile email groups")
OIDC_REDIRECT_URI = os.environ.get("OIDC_REDIRECT_URI", "")  # override; else derived per request
OIDC_GROUPS_CLAIM = os.environ.get("OIDC_GROUPS_CLAIM", "groups")
OIDC_USERNAME_CLAIM = os.environ.get("OIDC_USERNAME_CLAIM", "preferred_username")
OIDC_EMAIL_CLAIM = os.environ.get("OIDC_EMAIL_CLAIM", "email")
OIDC_GROUP_ROLES_RAW = os.environ.get("OIDC_GROUP_ROLES", "")
# Default role for an authenticated SSO user whose groups map to nothing. Empty = deny (the
# user authenticates but gets no role → every scoped route 403s). Defaults to 'user'.
OIDC_DEFAULT_ROLE = os.environ.get("OIDC_DEFAULT_ROLE", "user") or None
OIDC_TX_TTL = int(os.environ.get("OIDC_TX_TTL", "600"))  # login-transaction cookie lifetime
OIDC_TX_COOKIE = "oidc_tx"

# ── Scope model (chat-ui domain) ────────────────────────────────────────────
CHAT_USE = "chat:use"
CONVOS_READ = "convos:read"
CONVOS_WRITE = "convos:write"
MODELS_READ = "models:read"
USERS_MANAGE = "users:manage"

ROLES = RoleScopes(
    {
        "admin": {CHAT_USE, CONVOS_READ, CONVOS_WRITE, MODELS_READ, USERS_MANAGE},
        "user": {CHAT_USE, CONVOS_READ, CONVOS_WRITE, MODELS_READ},
    }
)

# Cookie signer — tamper-evident wrapper around the opaque session token. If SESSION_SECRET
# is unset we mint an ephemeral one (dev only; the boot check refuses this in production).
_secret = SESSION_SECRET or secrets.token_urlsafe(32)
_signer = URLSafeTimedSerializer(_secret, salt="chat-ui-session")
# Separate salt for the short-lived OIDC login-transaction cookie (state/nonce/PKCE verifier).
_tx_signer = URLSafeTimedSerializer(_secret, salt="chat-ui-oidc-tx")

_ph = PasswordHasher()
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")


class RateLimiter:
    """Sliding-window failed-attempt limiter, keyed by an opaque string (username+IP)."""

    def __init__(self, max_attempts: int, window: int):
        self.max = max_attempts
        self.window = window
        self._hits: dict[str, list[float]] = defaultdict(list)

    def _prune(self, key: str, now: float) -> None:
        self._hits[key] = [t for t in self._hits[key] if now - t < self.window]

    def blocked(self, key: str) -> bool:
        now = time.time()
        self._prune(key, now)
        return len(self._hits[key]) >= self.max

    def record_failure(self, key: str) -> None:
        self._hits[key].append(time.time())

    def reset(self, key: str) -> None:
        self._hits.pop(key, None)


_login_limiter = RateLimiter(LOGIN_RATE_LIMIT, LOGIN_RATE_WINDOW)


# ── Database ────────────────────────────────────────────────────────────────
def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                email         TEXT,
                password_hash TEXT,                       -- NULL for SSO accounts
                auth_source   TEXT NOT NULL DEFAULT 'local',  -- 'local' | 'oidc'
                oidc_sub      TEXT,
                role          TEXT NOT NULL DEFAULT 'user',   -- 'admin' | 'user'
                status        TEXT NOT NULL DEFAULT 'pending', -- 'pending'|'active'|'disabled'
                created_at    INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title      TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_convos_user ON conversations(user_id);
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,              -- 'user' | 'assistant'
                content         TEXT NOT NULL,              -- raw markdown (incl. citations)
                created_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_convo ON messages(conversation_id);
            """
        )


# ── Sessions ────────────────────────────────────────────────────────────────
def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with connect() as c:
        c.execute(
            "INSERT INTO sessions(token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
            (token, user_id, now, now + SESSION_TTL),
        )
    return token


def lookup_active_user(token: str) -> Optional[sqlite3.Row]:
    """Resolve a session token to its *active* user, or None (expired/disabled/unknown)."""
    now = int(time.time())
    with connect() as c:
        return c.execute(
            """SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
               WHERE s.token = ? AND s.expires_at > ? AND u.status = 'active'""",
            (token, now),
        ).fetchone()


def delete_session(token: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


def cookie_value(token: str) -> str:
    """Sign an opaque session token for the cookie (tamper-evident)."""
    return _signer.dumps(token)


def set_session_cookie(response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        cookie_value(token),
        max_age=SESSION_TTL,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(COOKIE_NAME)


# ── Local accounts ───────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return _ph.hash(pw)


def verify_password(stored_hash: Optional[str], pw: str) -> bool:
    if not stored_hash:
        return False
    try:
        return _ph.verify(stored_hash, pw)
    except Argon2Error:
        return False


def get_local_user(username: str) -> Optional[sqlite3.Row]:
    with connect() as c:
        return c.execute(
            "SELECT * FROM users WHERE username = ? AND auth_source = 'local'", (username,)
        ).fetchone()


def get_user(user_id: int) -> Optional[sqlite3.Row]:
    with connect() as c:
        return c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def count_users() -> int:
    with connect() as c:
        return c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]


def count_active_admins(exclude_id: Optional[int] = None) -> int:
    with connect() as c:
        return c.execute(
            "SELECT COUNT(*) AS n FROM users "
            "WHERE role = 'admin' AND status = 'active' AND id IS NOT ?",
            (exclude_id,),
        ).fetchone()["n"]


def create_local_user(username: str, password: str, email: Optional[str], role: str, status: str) -> int:
    now = int(time.time())
    with connect() as c:
        cur = c.execute(
            "INSERT INTO users(username, email, password_hash, auth_source, role, status, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (username, email, hash_password(password), "local", role, status, now),
        )
        return cur.lastrowid


def break_glass_match(username: str, password: str) -> bool:
    if not BREAK_GLASS_ENABLED:
        return False
    return secrets.compare_digest(username, BREAK_GLASS_USER) and secrets.compare_digest(
        password, BREAK_GLASS_PASSWORD
    )


def ensure_break_glass_user() -> sqlite3.Row:
    """Return the break-glass admin's user row, creating/repairing it if needed. Lets the
    break-glass login issue a normal session even when the user table is empty."""
    now = int(time.time())
    with connect() as c:
        c.execute(
            "INSERT INTO users(username, auth_source, role, status, created_at) "
            "VALUES (?, 'local', 'admin', 'active', ?) "
            "ON CONFLICT(username) DO UPDATE SET role='admin', status='active'",
            (BREAK_GLASS_USER, now),
        )
        return c.execute("SELECT * FROM users WHERE username = ?", (BREAK_GLASS_USER,)).fetchone()


def principal_for(user: sqlite3.Row) -> Principal:
    role = user["role"]
    scopes = ROLES.scopes_for_role(role) if role in ROLES else frozenset()
    method = AUTH_OIDC if user["auth_source"] == "oidc" else AUTH_LOCAL
    return Principal(subject=f"{user['auth_source']}:{user['id']}", scopes=scopes, auth_method=method)


# ── OIDC user provisioning ───────────────────────────────────────────────────
def _unique_username(c: sqlite3.Connection, base: str) -> str:
    """Sanitise an IdP-supplied name to our username rules and make it unique."""
    base = re.sub(r"[^A-Za-z0-9._-]", "", base or "")[:32]
    if len(base) < 3:
        base = (base + "user")[:32]
    candidate, i = base, 1
    while c.execute("SELECT 1 FROM users WHERE username = ?", (candidate,)).fetchone():
        suffix = str(i)
        candidate, i = base[: 32 - len(suffix)] + suffix, i + 1
    return candidate


def upsert_oidc_user(sub: str, username: str, email: Optional[str], role: str) -> sqlite3.Row:
    """Find-or-create the user row for an SSO identity (keyed by the IdP subject).

    On every login the role and email are refreshed from the IdP claims — the IdP's groups
    are authoritative for SSO accounts. A locally-disabled SSO account stays disabled.
    """
    now = int(time.time())
    with connect() as c:
        row = c.execute(
            "SELECT * FROM users WHERE oidc_sub = ? AND auth_source = 'oidc'", (sub,)
        ).fetchone()
        if row is None:
            uname = _unique_username(c, username)
            cur = c.execute(
                "INSERT INTO users(username, email, auth_source, oidc_sub, role, status, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (uname, email, "oidc", sub, role, "active", now),
            )
            uid = cur.lastrowid
        else:
            uid = row["id"]
            status = "disabled" if row["status"] == "disabled" else "active"
            c.execute("UPDATE users SET email = ?, role = ?, status = ? WHERE id = ?",
                      (email, role, status, uid))
        return c.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()


# ── Conversations + messages (per-user data) ─────────────────────────────────
def create_conversation(user_id: int, title: Optional[str]) -> sqlite3.Row:
    now = int(time.time())
    with connect() as c:
        cur = c.execute(
            "INSERT INTO conversations(user_id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (user_id, title, now, now),
        )
        return c.execute("SELECT * FROM conversations WHERE id = ?", (cur.lastrowid,)).fetchone()


def list_conversations_for(user_id: int) -> list[sqlite3.Row]:
    """A user's conversations, most-recently-updated first, with a message count."""
    with connect() as c:
        return c.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count "
            "FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id "
            "WHERE c.user_id = ? GROUP BY c.id ORDER BY c.updated_at DESC, c.id DESC",
            (user_id,),
        ).fetchall()


def get_conversation(user_id: int, conv_id: int) -> Optional[sqlite3.Row]:
    """A conversation, only if it belongs to this user (ownership is the data boundary)."""
    with connect() as c:
        return c.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user_id)
        ).fetchone()


def rename_conversation(conv_id: int, title: str) -> None:
    now = int(time.time())
    with connect() as c:
        c.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, now, conv_id))


def delete_conversation(conv_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))  # messages cascade (FK)


def list_messages(conv_id: int) -> list[sqlite3.Row]:
    with connect() as c:
        return c.execute(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()


def add_message(conv_id: int, role: str, content: str) -> sqlite3.Row:
    """Append a message and bump the conversation's updated_at. Reused by the B5 chat flow."""
    now = int(time.time())
    with connect() as c:
        cur = c.execute(
            "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (?,?,?,?)",
            (conv_id, role, content, now),
        )
        c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        return c.execute("SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)).fetchone()


# ── OIDC Relying-Party provider ──────────────────────────────────────────────
def _parse_group_roles(raw: str) -> dict[str, str]:
    """Parse ``"group-a:admin,group-b:user"`` into a dict (IdP group -> chat-ui role)."""
    out: dict[str, str] = {}
    for entry in (s.strip() for s in raw.split(",") if s.strip()):
        group, _, role = entry.partition(":")
        if group.strip() and role.strip():
            out[group.strip()] = role.strip()
    return out


@dataclass
class OIDCProvider:
    """A configured OIDC Relying Party: builds the auth-redirect, exchanges the code, and
    resolves a validated ID token to a (subject, username, email, role) identity."""

    client_id: str
    client_secret: str
    scopes: str
    authorization_endpoint: str
    token_endpoint: str
    validator: OIDCValidator
    username_claim: str
    email_claim: str
    groups_claim: str
    group_roles: dict[str, str]
    default_role: Optional[str]
    http_client: Optional[httpx.AsyncClient] = None  # injected in tests; else created per call

    def authorization_url(self, *, redirect_uri: str, state: str, nonce: str, code_challenge: str) -> str:
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": self.scopes,
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return f"{self.authorization_endpoint}?{urlencode(params)}"

    async def exchange_code(self, *, code: str, code_verifier: str, redirect_uri: str) -> str:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code_verifier": code_verifier,
        }
        client = self.http_client or httpx.AsyncClient(timeout=10.0)
        try:
            resp = await client.post(self.token_endpoint, data=data)
            resp.raise_for_status()
            body = resp.json()
        finally:
            if self.http_client is None:
                await client.aclose()
        id_token = body.get("id_token")
        if not id_token:
            raise OIDCError("token endpoint response contained no id_token")
        return id_token

    async def identity(self, id_token: str, *, nonce: str) -> tuple[str, str, Optional[str], Optional[str]]:
        claims = await self.validator.validate_id_token(id_token, nonce=nonce)
        sub = claims.get(self.validator.cfg.subject_claim) or claims.get("sub")
        if not sub:
            raise OIDCError("id_token has no subject claim")
        username = claims.get(self.username_claim) or claims.get(self.email_claim) or f"oidc-{sub}"
        email = claims.get(self.email_claim)
        groups = claims.get(self.groups_claim, [])
        if isinstance(groups, str):
            groups = [groups]
        return sub, username, email, self._role_for(groups)

    def _role_for(self, groups: list[str]) -> Optional[str]:
        mapped = [self.group_roles[g] for g in groups if g in self.group_roles]
        if "admin" in mapped:
            return "admin"
        if mapped:  # any other mapped group → plain user (chat-ui has a two-role model)
            return "user"
        return self.default_role


def oidc_configured() -> bool:
    return bool(OIDC_ENABLED and OIDC_ISSUER and OIDC_CLIENT_ID and OIDC_CLIENT_SECRET)


_oidc_provider: Optional[OIDCProvider] = None


async def build_oidc_provider() -> OIDCProvider:
    """Discover the IdP's endpoints and assemble the provider (network call — lazy, on first use
    so the IdP need not be reachable at boot). Tests assign `_oidc_provider` directly instead."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        doc = (await client.get(f"{OIDC_ISSUER.rstrip('/')}/.well-known/openid-configuration")).json()
    jwks_uri = doc["jwks_uri"]

    async def fetch() -> list[dict]:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(jwks_uri)
            r.raise_for_status()
            return r.json().get("keys", [])

    group_roles = _parse_group_roles(OIDC_GROUP_ROLES_RAW)
    cfg = OIDCConfig(
        issuer=OIDC_ISSUER,
        audience=OIDC_CLIENT_ID,  # ID tokens are audienced to the client
        jwks_uri=jwks_uri,
        groups_claim=OIDC_GROUPS_CLAIM,
        group_roles=group_roles,
        default_role=OIDC_DEFAULT_ROLE,
    )
    validator = OIDCValidator(cfg, ROLES, JWKSCache(fetch, ttl=600, min_refresh_interval=30))
    return OIDCProvider(
        client_id=OIDC_CLIENT_ID,
        client_secret=OIDC_CLIENT_SECRET,
        scopes=OIDC_SCOPES,
        authorization_endpoint=doc["authorization_endpoint"],
        token_endpoint=doc["token_endpoint"],
        validator=validator,
        username_claim=OIDC_USERNAME_CLAIM,
        email_claim=OIDC_EMAIL_CLAIM,
        groups_claim=OIDC_GROUPS_CLAIM,
        group_roles=group_roles,
        default_role=OIDC_DEFAULT_ROLE,
    )


async def get_oidc_provider() -> OIDCProvider:
    global _oidc_provider
    if _oidc_provider is None:
        _oidc_provider = await build_oidc_provider()
    return _oidc_provider


# ── Boot ────────────────────────────────────────────────────────────────────
def run_boot_checks() -> None:
    """Fail-closed startup validation. Refuses an unsafe production config; logs warnings."""
    if OIDC_ENABLED and not (OIDC_ISSUER and OIDC_CLIENT_ID and OIDC_CLIENT_SECRET):
        raise AuthConfigError(
            "OIDC_ENABLED but OIDC_ISSUER / OIDC_CLIENT_ID / OIDC_CLIENT_SECRET are not all set."
        )
    break_glass = BREAK_GLASS_ENABLED
    any_auth = LOCAL_LOGIN_ENABLED or OIDC_ENABLED or break_glass

    warnings = verify_boot_config(
        production=PRODUCTION,
        any_auth_enabled=any_auth,
        oidc_enabled=OIDC_ENABLED,
        break_glass_present=break_glass,
    )
    if PRODUCTION and not SESSION_SECRET:
        raise AuthConfigError(
            "SESSION_SECRET must be set in production (used to sign session cookies)."
        )
    for w in warnings:
        print(f"[chat-ui] WARN: {w}", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    run_boot_checks()
    yield


app = FastAPI(title="chat-ui", docs_url=None, redoc_url=None, lifespan=lifespan)

# Prometheus metrics at GET /metrics — open like /health (request counts/latencies only).
# The attach_principal middleware doesn't reject, and auth is enforced per-route, so /metrics
# is unauthenticated by design. Network-gate it.
Instrumentator().instrument(app).expose(app)

# Static SPA assets (app.js, styles.css, vendored marked + DOMPurify). The "/" route serves
# index.html with branding injected; everything else under /static is served verbatim.
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def attach_principal(request: Request, call_next):
    """Resolve the session cookie to a Principal (and user row) on request.state."""
    request.state.principal = None
    request.state.user = None
    raw = request.cookies.get(COOKIE_NAME)
    if raw:
        try:
            token = _signer.loads(raw, max_age=SESSION_TTL)
        except (BadSignature, SignatureExpired):
            token = None
        if token:
            user = lookup_active_user(token)
            if user is not None:
                request.state.principal = principal_for(user)
                request.state.user = user
    return await call_next(request)


# Content-Security-Policy: scripts/styles only from this origin (the vendored marked +
# DOMPurify are served from /static), which is the real defence-in-depth against XSS in the
# LLM-influenced markdown we render. Images are allowed from anywhere so page-image citations
# (served from the operator-set INGESTION_PUBLIC_URL) load. 'unsafe-inline' covers the small
# inline <style> block in index.html; no inline scripts are used.
_CSP = (
    "default-src 'self'; "
    "img-src 'self' data: http: https:; "
    "style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; "
    "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("Content-Security-Policy", _CSP)
    return resp


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/auth/me")
async def auth_me(request: Request):
    # current_principal raises 401 when there's no valid session.
    principal = current_principal(request)
    user = request.state.user
    return {**whoami_payload(principal), "username": user["username"], "role": user["role"]}


# ── Local auth: register / login / logout ────────────────────────────────────
class RegisterBody(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/api/auth/register")
async def register(body: RegisterBody):
    if not REGISTRATION_ENABLED:
        raise HTTPException(403, "Registration is disabled")
    username = body.username.strip()
    if not _USERNAME_RE.match(username):
        raise HTTPException(422, "Username must be 3-32 chars: letters, digits, . _ -")
    if len(body.password) < MIN_PASSWORD_LEN:
        raise HTTPException(422, f"Password must be at least {MIN_PASSWORD_LEN} characters")
    if get_local_user(username) is not None:
        raise HTTPException(409, "Username already taken")

    # First account ever bootstraps the system as an active admin.
    first = count_users() == 0
    role = "admin" if first else "user"
    status = "active" if (first or not REQUIRE_APPROVAL) else "pending"
    user_id = create_local_user(username, body.password, body.email, role, status)

    if status == "active":
        token = create_session(user_id)
        resp = JSONResponse({"status": "active", "role": role, "username": username})
        set_session_cookie(resp, token)
        return resp
    return JSONResponse(
        {"status": "pending", "message": "Account created; awaiting administrator approval."},
        status_code=202,
    )


@app.post("/api/auth/login")
async def login(request: Request, body: LoginBody):
    if not LOCAL_LOGIN_ENABLED and not break_glass_match(body.username, body.password):
        raise HTTPException(403, "Local login is disabled")

    client = request.client.host if request.client else "?"
    rl_key = f"{client}:{body.username}"
    if _login_limiter.blocked(rl_key):
        raise HTTPException(429, "Too many login attempts; try again later")

    # Break-glass first — works even if the user table is empty or local login is off.
    if break_glass_match(body.username, body.password):
        user = ensure_break_glass_user()
        token = create_session(user["id"])
        _login_limiter.reset(rl_key)
        resp = JSONResponse({"status": "active", "role": "admin", "username": user["username"]})
        set_session_cookie(resp, token)
        return resp

    user = get_local_user(body.username)
    if user is None or not verify_password(user["password_hash"], body.password):
        _login_limiter.record_failure(rl_key)
        raise HTTPException(401, "Invalid username or password")
    if user["status"] == "pending":
        raise HTTPException(403, "Account awaiting administrator approval")
    if user["status"] != "active":
        raise HTTPException(403, "Account is disabled")

    token = create_session(user["id"])
    _login_limiter.reset(rl_key)
    resp = JSONResponse({"status": "active", "role": user["role"], "username": user["username"]})
    set_session_cookie(resp, token)
    return resp


@app.post("/api/auth/logout")
async def logout(request: Request):
    raw = request.cookies.get(COOKIE_NAME)
    if raw:
        try:
            delete_session(_signer.loads(raw, max_age=SESSION_TTL))
        except (BadSignature, SignatureExpired):
            pass
    resp = JSONResponse({"status": "ok"})
    clear_session_cookie(resp)
    return resp


# ── OIDC single-sign-on (Relying Party flow) ─────────────────────────────────
def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _callback_redirect_uri(request: Request) -> str:
    return OIDC_REDIRECT_URI or str(request.url_for("oidc_callback"))


@app.get("/api/auth/oidc/login")
async def oidc_login(request: Request):
    if not oidc_configured():
        raise HTTPException(404, "OIDC sign-in is not enabled")
    provider = await get_oidc_provider()

    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)  # PKCE code_verifier
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())  # S256
    redirect_uri = _callback_redirect_uri(request)

    url = provider.authorization_url(
        redirect_uri=redirect_uri, state=state, nonce=nonce, code_challenge=challenge
    )
    # The login transaction (state/nonce/verifier/redirect) rides in a short-lived signed,
    # httpOnly cookie rather than server state — no table, and it's bound to this browser.
    tx = _tx_signer.dumps(
        {"state": state, "nonce": nonce, "verifier": verifier, "redirect_uri": redirect_uri}
    )
    resp = RedirectResponse(url, status_code=302)
    resp.set_cookie(
        OIDC_TX_COOKIE, tx, max_age=OIDC_TX_TTL,
        httponly=True, secure=COOKIE_SECURE, samesite="lax",
    )
    return resp


@app.get("/api/auth/oidc/callback", name="oidc_callback")
async def oidc_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    if not oidc_configured():
        raise HTTPException(404, "OIDC sign-in is not enabled")
    if error:
        raise HTTPException(400, f"Identity provider returned an error: {error}")
    if not code or not state:
        raise HTTPException(400, "Missing authorization code or state")

    raw = request.cookies.get(OIDC_TX_COOKIE)
    if not raw:
        raise HTTPException(400, "Missing or expired login transaction; please sign in again")
    try:
        tx = _tx_signer.loads(raw, max_age=OIDC_TX_TTL)
    except (BadSignature, SignatureExpired):
        raise HTTPException(400, "Invalid or expired login transaction; please sign in again")
    if not secrets.compare_digest(state, tx["state"]):
        raise HTTPException(400, "State mismatch")  # CSRF guard

    provider = await get_oidc_provider()
    try:
        id_token = await provider.exchange_code(
            code=code, code_verifier=tx["verifier"], redirect_uri=tx["redirect_uri"]
        )
        sub, username, email, role = await provider.identity(id_token, nonce=tx["nonce"])
    except OIDCError as exc:
        raise HTTPException(401, f"OIDC login failed: {exc}")
    except httpx.HTTPError:
        raise HTTPException(502, "Could not reach the identity provider")

    if role is None:
        raise HTTPException(403, "Your account is not authorised for this application")

    user = upsert_oidc_user(sub, username, email, role)
    if user["status"] != "active":
        raise HTTPException(403, "Account is disabled")

    token = create_session(user["id"])  # fresh session on every SSO login
    resp = RedirectResponse("/", status_code=302)
    set_session_cookie(resp, token)
    resp.delete_cookie(OIDC_TX_COOKIE)
    return resp


@app.get("/api/auth/config")
async def auth_config():
    """Unauthenticated: lets the SPA decide which sign-in options to render."""
    return {
        "brand_name": BRAND_NAME,
        "brand_primary_color": BRAND_PRIMARY_COLOR,
        "local_login": LOCAL_LOGIN_ENABLED,
        "registration_enabled": REGISTRATION_ENABLED and LOCAL_LOGIN_ENABLED,
        "oidc": {"enabled": oidc_configured(), "login_path": "/api/auth/oidc/login"},
        # Whether ai-agent has a web-search provider configured — the SPA only shows the
        # per-conversation "Search the web" toggle when this is true.
        "web_search": bool((await _agent_capabilities()).get("web_search", False)),
    }


# ── Admin: user management (requires users:manage) ───────────────────────────
class RoleBody(BaseModel):
    role: str


_admin = Depends(require_scope(USERS_MANAGE))


@app.get("/api/admin/users", dependencies=[_admin])
async def admin_list_users():
    with connect() as c:
        rows = c.execute(
            "SELECT id, username, email, auth_source, role, status, created_at "
            "FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def _require_user(user_id: int) -> sqlite3.Row:
    user = get_user(user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    return user


@app.post("/api/admin/users/{user_id}/approve", dependencies=[_admin])
async def admin_approve(user_id: int):
    _require_user(user_id)
    with connect() as c:
        c.execute("UPDATE users SET status = 'active' WHERE id = ? AND status = 'pending'", (user_id,))
    return {"status": "active"}


@app.post("/api/admin/users/{user_id}/disable", dependencies=[_admin])
async def admin_disable(user_id: int):
    user = _require_user(user_id)
    if user["role"] == "admin" and user["status"] == "active" and count_active_admins(exclude_id=user_id) == 0:
        raise HTTPException(409, "Cannot disable the last active admin")
    with connect() as c:
        c.execute("UPDATE users SET status = 'disabled' WHERE id = ?", (user_id,))
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))  # force logout
    return {"status": "disabled"}


@app.post("/api/admin/users/{user_id}/role", dependencies=[_admin])
async def admin_set_role(user_id: int, body: RoleBody):
    if body.role not in ROLES:
        raise HTTPException(422, f"Unknown role: {body.role}")
    user = _require_user(user_id)
    demoting_admin = user["role"] == "admin" and body.role != "admin"
    if demoting_admin and user["status"] == "active" and count_active_admins(exclude_id=user_id) == 0:
        raise HTTPException(409, "Cannot demote the last active admin")
    with connect() as c:
        c.execute("UPDATE users SET role = ? WHERE id = ?", (body.role, user_id))
    return {"role": body.role}


# ── Conversations (per-user; convos:read / convos:write) ─────────────────────
class ConversationBody(BaseModel):
    title: Optional[str] = None


class RenameBody(BaseModel):
    title: str


class MessageBody(BaseModel):
    role: str
    content: str


_convos_read = Depends(require_scope(CONVOS_READ))
_convos_write = Depends(require_scope(CONVOS_WRITE))


def _current_user_id(request: Request) -> int:
    # Guaranteed present: the scope dependency 401s before the route runs without a principal.
    return request.state.user["id"]


def _owned_or_404(request: Request, conv_id: int) -> sqlite3.Row:
    conv = get_conversation(_current_user_id(request), conv_id)
    if conv is None:
        # 404 (not 403) so a conversation's existence never leaks to a non-owner.
        raise HTTPException(404, "Conversation not found")
    return conv


def _conv_summary(row: sqlite3.Row, message_count: int) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "message_count": message_count,
    }


@app.post("/api/conversations", dependencies=[_convos_write])
async def create_conversation_route(request: Request, body: ConversationBody):
    title = (body.title or "").strip() or None
    row = create_conversation(_current_user_id(request), title)
    return _conv_summary(row, 0)


@app.get("/api/conversations", dependencies=[_convos_read])
async def list_conversations_route(request: Request):
    rows = list_conversations_for(_current_user_id(request))
    return [_conv_summary(r, r["message_count"]) for r in rows]


@app.get("/api/conversations/{conv_id}", dependencies=[_convos_read])
async def get_conversation_route(request: Request, conv_id: int):
    conv = _owned_or_404(request, conv_id)
    messages = [dict(m) for m in list_messages(conv_id)]
    return {**_conv_summary(conv, len(messages)), "messages": messages}


@app.patch("/api/conversations/{conv_id}", dependencies=[_convos_write])
async def rename_conversation_route(request: Request, conv_id: int, body: RenameBody):
    _owned_or_404(request, conv_id)
    title = body.title.strip()
    if not title:
        raise HTTPException(422, "Title must not be empty")
    rename_conversation(conv_id, title)
    return {"id": conv_id, "title": title}


@app.delete("/api/conversations/{conv_id}", dependencies=[_convos_write])
async def delete_conversation_route(request: Request, conv_id: int):
    _owned_or_404(request, conv_id)
    delete_conversation(conv_id)
    return {"status": "deleted"}


@app.post("/api/conversations/{conv_id}/messages", dependencies=[_convos_write])
async def add_message_route(request: Request, conv_id: int, body: MessageBody):
    _owned_or_404(request, conv_id)
    if body.role not in ("user", "assistant"):
        raise HTTPException(422, "role must be 'user' or 'assistant'")
    if not body.content.strip():
        raise HTTPException(422, "content must not be empty")
    return dict(add_message(conv_id, body.role, body.content))


# ── Chat (proxy to ai-agent /v1) + models ────────────────────────────────────
# ai-agent assembles the whole answer (with its Sources/citation markdown appended) before it
# replays it, and only the non-streaming response carries structured sources. So chat-ui calls
# it with stream=False, persists the turn, then replays the saved answer to the browser as SSE.
# Everything fallible (the LLM call, persistence) happens before the stream starts, so failures
# surface as a normal HTTP status rather than a half-sent stream.
class ChatBody(BaseModel):
    conversation_id: int
    content: str
    model: Optional[str] = None
    web_search: bool = False   # per-conversation opt-in; forwarded to ai-agent (default grounded)


_default_model: Optional[str] = None


async def _agent_chat(messages: list[dict], model: str, web_search: bool = False) -> dict:
    payload = {"model": model, "messages": messages, "stream": False, "web_search": web_search}
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(f"{AI_AGENT_URL}/chat/completions", json=payload, headers=_SERVICE_AUTH)
        resp.raise_for_status()
        return resp.json()


async def _agent_models() -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{AI_AGENT_URL}/models", headers=_SERVICE_AUTH)
        resp.raise_for_status()
        return resp.json()


async def _agent_capabilities() -> dict:
    """ai-agent feature flags (e.g. whether web search has a configured provider). Best-effort:
    returns {} on any error so the config path never hard-depends on ai-agent being reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{AI_AGENT_URL}/capabilities", headers=_SERVICE_AUTH)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {}


async def resolve_model(provided: Optional[str]) -> str:
    """The model to ask for: the client's choice, else the configured default, else the first
    model ai-agent advertises (discovered once and cached)."""
    global _default_model
    if provided:
        return provided
    if DEFAULT_MODEL:
        return DEFAULT_MODEL
    if _default_model is None:
        models = (await _agent_models()).get("data", [])
        _default_model = models[0]["id"] if models else "default"
    return _default_model


def _derive_title(text: str) -> str:
    return " ".join(text.split())[:60] or "New conversation"


# ai-agent appends these blocks to its answers (see ai-agent format_citations / format_sources).
# They're rendering output — URLs, page refs, image links, verbatim quotes — so we keep them in
# the stored/displayed message but strip them out of the history we re-feed the model as input.
_CITATION_MARKERS = ("\n\n**Verified quotes:**", "\n\n---\n**Sources:**")


def _strip_citation_trailer(content: str) -> str:
    cut = len(content)
    for marker in _CITATION_MARKERS:
        idx = content.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return content[:cut].rstrip()


def _history_for_agent(conv_id: int) -> list[dict]:
    out = []
    for m in list_messages(conv_id):
        text = _strip_citation_trailer(m["content"]) if m["role"] == "assistant" else m["content"]
        out.append({"role": m["role"], "content": text})
    return out


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/chat", dependencies=[Depends(require_scope(CHAT_USE))])
async def chat(request: Request, body: ChatBody):
    conv = get_conversation(_current_user_id(request), body.conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    content = body.content.strip()
    if not content:
        raise HTTPException(422, "Message must not be empty")

    add_message(body.conversation_id, "user", content)
    if not conv["title"]:  # name a fresh conversation after its first message
        rename_conversation(body.conversation_id, _derive_title(content))

    history = _history_for_agent(body.conversation_id)
    model = await resolve_model(body.model)
    try:
        agent_json = await _agent_chat(history, model, web_search=body.web_search)
        answer = agent_json["choices"][0]["message"]["content"]
    except httpx.HTTPError:
        raise HTTPException(502, "The AI service is unavailable")
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, "The AI service returned an unexpected response")

    saved = add_message(body.conversation_id, "assistant", answer)

    async def replay():
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield _sse({"delta": word if i == len(words) - 1 else word + " "})
            if STREAM_DELAY:
                await asyncio.sleep(STREAM_DELAY)
        yield _sse({"done": True, "message_id": saved["id"], "conversation_id": body.conversation_id})

    return StreamingResponse(replay(), media_type="text/event-stream")


@app.get("/api/models", dependencies=[Depends(require_scope(MODELS_READ))])
async def models():
    try:
        data = await _agent_models()
    except httpx.HTTPError:
        raise HTTPException(502, "The AI service is unavailable")
    return [{"id": m.get("id")} for m in data.get("data", [])]


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__BRAND__", BRAND_NAME).replace("__COLOR__", BRAND_PRIMARY_COLOR))
