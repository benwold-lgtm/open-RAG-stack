"""chat-ui — first-party multi-user chat service (Open-WebUI replacement).

B1 (this file at the skeleton stage): FastAPI + SQLite + signed server-side sessions, the
rag_auth scope seam, a branded SPA shell, an open /health, and a fail-closed boot check.
Local login (B2), OIDC (B3), conversations (B4), chat streaming (B5) and the full SPA (B6)
build on top of this.

The session model: a login (later) creates a row in `sessions` and sets a signed, httpOnly,
Secure, SameSite cookie carrying the session token. Every request resolves that cookie back to
a Principal (via rag_auth) so routes guard on scopes, never role strings.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

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
from rag_auth.oidc import build_oidc_config

# ── Config (env) ────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("CHAT_UI_DB", "/data/chat_ui.db")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
SESSION_TTL = int(os.environ.get("SESSION_TTL", str(7 * 24 * 3600)))  # 7 days
COOKIE_NAME = "chat_session"
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() in ("1", "true", "yes")
PRODUCTION = os.environ.get("ENVIRONMENT", "production").lower() == "production"
LOCAL_LOGIN_ENABLED = os.environ.get("LOCAL_LOGIN_ENABLED", "true").lower() in ("1", "true", "yes")
AI_AGENT_URL = os.environ.get("AI_AGENT_URL", "http://ai-agent:8000/v1")  # used in B5
BRAND_NAME = os.environ.get("BRAND_NAME", "Open RAG Chat")
BRAND_PRIMARY_COLOR = os.environ.get("BRAND_PRIMARY_COLOR", "#2563eb")

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
    """Sign an opaque session token for the cookie (tamper-evident). Login/logout in B2 set
    and clear the cookie itself."""
    return _signer.dumps(token)


def principal_for(user: sqlite3.Row) -> Principal:
    role = user["role"]
    scopes = ROLES.scopes_for_role(role) if role in ROLES else frozenset()
    method = AUTH_OIDC if user["auth_source"] == "oidc" else AUTH_LOCAL
    return Principal(subject=f"{user['auth_source']}:{user['id']}", scopes=scopes, auth_method=method)


# ── Boot ────────────────────────────────────────────────────────────────────
def run_boot_checks() -> None:
    """Fail-closed startup validation. Refuses an unsafe production config; logs warnings."""
    oidc_cfg = build_oidc_config()  # raises if OIDC_ENABLED but misconfigured
    oidc_enabled = oidc_cfg is not None
    break_glass = bool(os.environ.get("BREAK_GLASS_ADMIN_KEY"))
    any_auth = LOCAL_LOGIN_ENABLED or oidc_enabled or break_glass

    warnings = verify_boot_config(
        production=PRODUCTION,
        any_auth_enabled=any_auth,
        oidc_enabled=oidc_enabled,
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


@app.middleware("http")
async def attach_principal(request: Request, call_next):
    """Resolve the session cookie to a Principal on request.state (None if no valid session)."""
    request.state.principal = None
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
    return await call_next(request)


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/auth/me")
async def auth_me(request: Request):
    # current_principal raises 401 when there's no valid session.
    return whoami_payload(current_principal(request))


# A scope-guarded placeholder proving the seam end-to-end; real conversation routes land in B4.
@app.get("/api/conversations", dependencies=[Depends(require_scope(CONVOS_READ))])
async def list_conversations():
    return []


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(SHELL)


# ── SPA shell (B1 placeholder — full vanilla SPA lands in B6) ────────────────
SHELL = (
    r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__BRAND__</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f1f5f9;color:#1e293b;min-height:100vh;
display:flex;align-items:center;justify-content:center}
.card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.12);padding:2rem 2.4rem;text-align:center;max-width:420px}
.dot{width:42px;height:42px;border-radius:9px;background:__COLOR__;margin:0 auto 1rem}
h1{font-size:1.15rem;font-weight:600;margin-bottom:.4rem}
p{font-size:.85rem;color:#64748b;line-height:1.6}
</style>
</head>
<body>
<div class="card">
  <div class="dot"></div>
  <h1>__BRAND__</h1>
  <p>Service is running. Sign-in and chat arrive in the next milestones.</p>
</div>
</body>
</html>"""
    .replace("__BRAND__", BRAND_NAME)
    .replace("__COLOR__", BRAND_PRIMARY_COLOR)
)
