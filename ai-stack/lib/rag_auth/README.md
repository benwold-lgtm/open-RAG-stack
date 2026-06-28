# rag_auth

Shared authentication/authorization for open-RAG-stack services. One small, reusable module
so every service (chat-ui first; later rag-admin, ai-agent) shares **one** identity model
instead of each rolling its own.

Built from a portable, battle-tested blueprint. The five load-bearing ideas:

1. **Authorize on scopes, never role strings.** A request resolves to a
   `Principal{subject, scopes, auth_method}`; every route guards on **one scope**. Roles are
   named bundles of scopes in a single `RoleScopes` table — adding a role is a data change,
   never a route change.
2. **One `authenticate()` seam.** Every mechanism resolves a credential to a `Principal` (or
   `None`). Swapping/adding a mechanism touches only the seam.
3. **Composite auth with break-glass:** `OIDC JWT → static key → 401`. An OIDC failure falls
   through to env-configured **break-glass** keys, so OIDC fails *closed* while break-glass
   keeps working through an IdP outage (or a broken user table).
4. **The service owns `group → role → scopes`; the UI mirrors it** via `/auth/me`. The IdP
   only asserts group membership.
5. **OIDC hardening is non-negotiable:** asymmetric-only algorithm allow-list, JWKS bounded
   cache + rate-limited unknown-kid refetch, full `iss`/`aud`/`exp` checks, never trust a key
   in the token.

This module is **domain-agnostic** — it ships no scope names. The consumer defines its scopes
and role map.

## Install (into a service image)

Path-installed; no private index. In a service Dockerfile:

```dockerfile
COPY ai-stack/lib/rag_auth /tmp/rag_auth
RUN pip install --no-cache-dir /tmp/rag_auth
```

(The Docker/Compose build context must include `ai-stack/lib/`.) For local dev:
`pip install -e ai-stack/lib/rag_auth[test]`.

## Quickstart — API-style service (bearer tokens)

```python
from fastapi import APIRouter, Depends, FastAPI
from rag_auth import (
    RoleScopes, build_static_authenticator, build_oidc_validator,
    CompositeAuthenticator, make_authenticate_request, require_scope,
    make_whoami_router, verify_boot_config,
)

ROLES = RoleScopes({
    "admin": {"things:read", "things:write", "users:manage"},
    "user":  {"things:read"},
})

static = build_static_authenticator(ROLES)              # break-glass keys from env
oidc   = build_oidc_validator(ROLES)                    # None unless OIDC_ENABLED
authn  = CompositeAuthenticator(static=static, oidc=oidc)

verify_boot_config(                                     # fail-closed at startup
    production=True,
    any_auth_enabled=authn.enabled,
    oidc_enabled=oidc is not None,
    break_glass_present=static.enabled,
)

app = FastAPI()
protected = APIRouter(dependencies=[Depends(make_authenticate_request(authn))])

@protected.get("/things", dependencies=[Depends(require_scope("things:read"))])
async def list_things(): ...

app.include_router(protected)
app.include_router(make_whoami_router(make_authenticate_request(authn)))  # GET /auth/me
```

## Quickstart — cookie-session app (e.g. chat-ui as a BFF)

A session app authenticates once at login (local password **or** OIDC), stores a server-side
session, and resolves each request from the cookie. It populates `request.state.principal`
itself and reuses `require_scope` / `current_principal`:

```python
from rag_auth import Principal, AUTH_LOCAL, require_scope, whoami_payload

@app.middleware("http")
async def attach_principal(request, call_next):
    user = lookup_session(request)            # your session store
    if user:
        request.state.principal = Principal(
            subject=f"local:{user.id}",
            scopes=ROLES.scopes_for_role(user.role),
            auth_method=AUTH_LOCAL,
        )
    return await call_next(request)

# routes still guard on scopes:
@app.post("/api/conversations", dependencies=[Depends(require_scope("convos:write"))])
async def create_convo(): ...

# /api/auth/me returns the same contract shape as API services:
@app.get("/api/auth/me")
async def me(request): return whoami_payload(current_principal(request))
```

The OIDC **login flow** (Authorization-Code + PKCE, state/nonce, session rotation) lives in
the consuming app (chat-ui Effort B3). `rag_auth.OIDCValidator` is reused there to validate
the ID token returned by the code exchange (construct it with `audience = client_id`); for
incoming access tokens, validate with `audience = service`.

## Config reference (environment)

Break-glass / static keys:

| Var | Effect |
|---|---|
| `BREAK_GLASS_ADMIN_KEY` | Bearer token that authenticates as the admin role. Keep set in prod. |
| `BREAK_GLASS_VIEWER_KEY` | Read-only break-glass token (if a viewer role exists). |
| `RAG_AUTH_KEYS` | Extra machine identities: `name:role:token,name:role:token`. |

OIDC (all optional; disabled unless `OIDC_ENABLED`):

| Var | Default | Effect |
|---|---|---|
| `OIDC_ENABLED` | off | `1`/`true`/`yes` to enable. |
| `OIDC_ISSUER` | — | IdP issuer URL (Entra/Okta/Google/Keycloak…). Required. |
| `OIDC_AUDIENCE` | — | Expected access-token `aud` (this service). Required. |
| `OIDC_JWKS_URI` | discovered | Set explicitly when air-gapped / discovery blocked. |
| `OIDC_GROUPS_CLAIM` | `groups` | Claim holding group membership. |
| `OIDC_SUBJECT_CLAIM` | `sub` | Claim used as the subject. |
| `OIDC_ALGORITHMS` | `RS256` | Comma list; **asymmetric only** (HS*/none refused at boot). |
| `OIDC_LEEWAY` | `60` | Clock-skew seconds. |
| `OIDC_JWKS_CACHE_TTL` | `600` | JWKS cache lifetime (seconds). |
| `OIDC_JWKS_MIN_REFRESH` | `30` | Min seconds between unknown-kid refetches. |
| `OIDC_GROUP_ROLES` | — | `group-a:admin,group-b:user`. |
| `OIDC_DEFAULT_ROLE` | none | Role for an authenticated user with no mapped group. Unset ⇒ zero scopes (default-deny). |

## Security checklist — boundary of responsibility

`rag_auth` **enforces** (covered + tested):

- [x] OIDC signature checked against JWKS; `iss`/`aud`/`exp` required; bounded clock skew.
- [x] Asymmetric-only algorithm allow-list; `none` and `HS*` refused at config time.
- [x] `kid` matched to a published JWKS key; a key embedded in the token is never trusted.
- [x] JWKS bounded-TTL cache + rate-limited unknown-kid refetch; cached keys served through
      an IdP outage.
- [x] OIDC fails **closed**, break-glass keys **open** only to themselves.
- [x] Constant-time comparison of static keys (`hmac.compare_digest`).
- [x] Fail-closed at boot (`verify_boot_config`): refuse no-auth in prod; warn on OIDC with no
      break-glass.
- [x] No-mapped-group ⇒ authenticated with zero (or minimal default-role) scopes, not a 401 —
      the audit records *who* was denied.

The **consuming service** must still do (Effort B / deployment):

- [ ] Authorization-Code **+ PKCE (S256)**, single-use `state`/`nonce`, session-id rotation on
      login (the RP login flow).
- [ ] Keep secrets/tokens **server-side**; never in `localStorage`/URL.
- [ ] Local-account password hashing (bcrypt/argon2), login rate-limiting, CSRF token on
      state-changing requests, secure cookie flags.
- [ ] Run the issuer/JWKS URL through an **egress/SSRF** policy.
- [ ] **Audit** subject + outcome on privileged actions and on 401/403 denials.

## Testing

```
pip install -e .[test]
pytest                       # 49 tests; fully offline (locally-signed RSA, no network IdP)
```

The OIDC suite signs its own tokens with an in-process RSA key and seeds the JWKS cache, so it
covers valid / expired / wrong-aud / wrong-iss / `none` / `HS256` / wrong-key / unknown-kid /
no-group, the rate-limited refetch, the outage path, and IdP-down break-glass — with no IdP.
