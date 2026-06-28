"""chat-ui — first-party multi-user chat service (Open-WebUI replacement).

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

import os
import re
import secrets
import sqlite3
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
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


# ── Boot ────────────────────────────────────────────────────────────────────
def run_boot_checks() -> None:
    """Fail-closed startup validation. Refuses an unsafe production config; logs warnings."""
    oidc_cfg = build_oidc_config()  # raises if OIDC_ENABLED but misconfigured
    oidc_enabled = oidc_cfg is not None
    break_glass = BREAK_GLASS_ENABLED
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
