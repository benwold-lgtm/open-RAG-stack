# Auth + Chat-UI — Build Plan

**Status:** Approved to build (spec-first complete).
**Last updated:** 2026-06-28.
**Companion docs:** product spec in [`CHAT-UI-SPEC.md`](CHAT-UI-SPEC.md); phase tracking in
[`ENHANCEMENT-PLAN.md`](ENHANCEMENT-PLAN.md) (this is **Phase 8**).

Two efforts, sequenced **auth-core first, one pass** (no local-then-OIDC rework):

- **Effort A — `rag_auth`**, a small **shared, reusable** auth module (the portable
  blueprint, adapted to this stack). Built and offline-tested with no UI.
- **Effort B — `chat-ui`**, a first-party multi-user chat service that **replaces Open-WebUI**
  and consumes `rag_auth`.
- **Effort C — later/optional** consumers: retrofit `rag-admin` off Basic Auth; guard
  `ai-agent`; true token streaming.

---

## Design decisions carried in (the load-bearing five, adapted)

1. **Authorize on scopes, never role strings.** A request resolves to a
   `Principal{subject, scopes, auth_method}`; every route guards on **one scope**. Roles are
   named bundles of scopes in a single table — adding a role is a data change, never a route
   change. *Scopes from day one* is what lets us skip a second auth pass.
2. **One `authenticate()` seam → `Principal`.** Swapping/adding a mechanism (local ↔ OIDC)
   touches only that seam; routes and audit are untouched.
3. **Composite auth with break-glass:** OIDC JWT → local → 401. A **break-glass admin**
   credential (env-configured, independent of OIDC *and* of the user DB) always works, so an
   IdP outage — or a broken user table — never locks everyone out. OIDC **fails closed**;
   break-glass **fails open only to itself**.
4. **The service owns `group → role → scopes`; the UI mirrors it** via a `/auth/me` whoami
   that returns scopes. The IdP only asserts group membership. UI and API can't drift.
5. **OIDC hardening is non-negotiable:** asymmetric-only algorithm allow-list (reject
   `none`/`HS*`), JWKS bounded-TTL cache + rate-limited unknown-kid refetch, full
   `iss`/`aud`/`exp` claim checks, never trust a key embedded in the token, PKCE + single-use
   `state`/`nonce`, session-id rotation on login, **fail-closed at boot**.

### One reconciliation vs the original 3-tier blueprint

The blueprint separates **Service/API (authorization point)** · **BFF (session + login +
relay)** · **SPA**. `chat-ui` **owns the resources it protects** (users, conversations,
messages), so we **collapse Tier 1 + Tier 2 into `chat-ui`**:

- `chat-ui` runs the OIDC Authorization-Code + PKCE flow (BFF role), maps groups → scopes and
  enforces them (service role), and mints its **own** server-side session.
- **No token passthrough/relay.** `ai-agent` stays behind the trust boundary as a plain
  inference backend — *not* a per-user authorization point. (A Tier-1 scope guard can be added
  to `ai-agent` later **iff** it's ever exposed as a standalone API — Effort C.)
- **"Local" = real accounts + break-glass.** Human users need real local accounts
  (registration, per-user history) → SQLite `users`. OIDC users get an **auto-provisioned row
  for data ownership**, but their **scopes are derived fresh from group claims each login**
  (persist identity, not scopes). The env break-glass admin sits on top of both.

---

## Scope model (chat-ui domain)

```
chat:use         send chat turns / use the assistant
convos:read      read own conversations + messages
convos:write     create / rename / delete own conversations
models:read      list available models
users:manage     admin: approve / disable / set-role
settings:manage  admin: branding / auth config (future)
```

| Role | Scopes |
|---|---|
| `admin` | all |
| `user`  | `chat:use, convos:read, convos:write, models:read` |

Per-user data isolation is still enforced in every query (`WHERE user_id = :me`); scopes gate
*capability*, ownership gates *rows*.

---

## Effort A — `rag_auth` shared module

**Location:** `ai-stack/lib/rag_auth/` — an installable package (`pyproject.toml`). Each
consuming service's Dockerfile does `COPY ai-stack/lib/rag_auth` then `pip install ./rag_auth`
(compose/CI build context must include `lib/`). No vendor IdP SDKs — PyJWT[crypto] + httpx.

PR-sized steps (each independently testable):

- **A1 — Scope/role model.** `rbac.py`: scope constants, `ROLE_SCOPES`, `Principal`,
  `scopes_for_role`. *Verify:* unit tests — role→scope bundles correct; unknown role raises.
- **A2 — Local/static authenticator + FastAPI seam.** `authenticator.py`
  (`Authenticator`, `build_static_authenticator` incl. the **break-glass admin** from env);
  `deps.py` (`authenticate_request` router dep, `require_scope(scope)` route-dep factory);
  **fail-closed boot** (refuse to start in prod with no auth configured; warn if OIDC enabled
  with no break-glass). *Verify:* viewer can read not write (403); missing/garbage token (401);
  admin passes; boot refuses with zero auth.
- **A3 — `/auth/me` whoami helper.** Returns `{subject, scopes, auth_method}`. Tiny but it's
  the UI contract — land it early. *Verify:* returns subject+scopes; 401 unauthenticated.
- **A4 — OIDC validator + composite authenticator.** `oidc.py` (`OIDCValidator`: JWKS cache w/
  bounded TTL + rate-limited unknown-kid refetch, asymmetric-only alg allow-list, `iss`/`aud`/
  `exp` checks, group→scope mapping), `CompositeAuthenticator` (OIDC → local → 401).
  *Verify (fully offline):* sign tokens with a local RSA key + seed the JWKS cache — cover
  valid / expired / wrong-aud / wrong-iss / bad-alg / unknown-kid / no-mapped-group, the
  rate-limited refresh, and **IdP-down break-glass** (OIDC 401 but the break-glass key still
  authenticates).
- **A5 — Threat-model pass.** Walk the security checklist (I1–I4 boundaries) against the
  implementation before any prod enablement.

**Exit criteria for Effort A:** `pip install ./rag_auth` works; full offline test suite green
including the IdP-down break-glass case; no UI required to validate.

---

## Effort B — `chat-ui` (replaces Open-WebUI), consuming `rag_auth`

**Location:** `ai-stack/services/chat-ui/` (`main.py`, `requirements.txt`, `Dockerfile`,
`static/` with vendored `marked` + `DOMPurify`). Takes Open-WebUI's slot (`:3001` /
NodePort `30080`); talks to `http://ai-agent:8000/v1`.

- **B1 — Service skeleton + DB + sessions.** FastAPI + SQLite (`users`, `sessions`,
  `conversations`, `messages`); signed httpOnly+Secure+SameSite session cookie; serve the SPA
  shell; `/health` (open). Wire the `rag_auth` seam + fail-closed boot. *Verify:* health open,
  every other route 401 without a session; boot refuses with no auth.
- **B2 — Local accounts + admin API.** register → `pending` → admin approve → `active`;
  first-user becomes `admin`+`active`; bcrypt/argon2; login/logout; login rate-limit;
  `/api/auth/me` (scopes); env **break-glass admin**; admin user-management
  (approve/disable/role). *Verify:* bootstrap admin; pending user blocked until approved;
  break-glass logs in with the DB user table emptied.
- **B3 — OIDC RP flow.** `/api/auth/oidc/login` (PKCE + single-use state/nonce),
  `/api/auth/oidc/callback` (state check, server-side code exchange, ID-token validation via
  `rag_auth`, session rotation, **auto-provision** user row, **derive scopes from group
  claims**); `/api/auth/config` (`oidc_enabled`, `local_login`). Local-login route stays
  reachable when OIDC is on. *Verify (offline, stubbed IdP):* state-mismatch → 400; happy path
  provisions a user and sets a session; unmapped group → minimal `user` role (see open
  decision); local login still works with OIDC enabled.
- **B4 — Conversations.** CRUD + per-user persistence; store assistant **raw markdown**
  (citation blocks included). *Verify:* user A cannot see user B's conversations.
- **B5 — Chat proxy + streaming.** `POST /api/conversations/{id}/chat` → proxy `ai-agent`
  with `stream:true` (**existing pseudo-streaming — zero `ai-agent` change**, citations already
  ordered), persist both turns; client renders markdown + citations/images via vendored
  `marked` + `DOMPurify`. *Verify:* a real RAG answer streams in and persists; reload renders
  identically; sources/images appear after the answer.
- **B6 — SPA + branding + admin screen.** login/register, sidebar (convo list, new chat, user
  menu), thread + streaming composer w/ stop, model dropdown; **scope-gated affordances** read
  from `/auth/me`; admin user table; `BRAND_NAME/LOGO/PRIMARY_COLOR/FAVICON` injected at render
  time. *Verify:* a non-admin never sees admin controls; rebrand by setting four env vars, no
  code change.
- **B7 — Wiring.** `ai-stack/charts/chat-ui/` (Deployment + Service NodePort 30080, env +
  branding + auth-secret wiring, mirroring existing charts); `.github/workflows/
  build-chat-ui.yml`; `docker-compose.yml` swap `open-webui` → `chat-ui`; README + architecture
  diagram + services table; remove the Open-WebUI license note. Stand up **alongside**
  Open-WebUI first.
- **B8 — Hardening + cutover.** Security-checklist pass; e2e on the GPU node; decommission
  Open-WebUI (image, env, volume).

---

## Effort C — later / optional

- **`rag-admin` onto `rag_auth`** — replace its standalone Basic Auth with the shared seam so
  the whole stack shares one identity model (single sign-on across admin + chat).
- **Guard `ai-agent`** — add the Tier-1 `require_scope` seam + per-user identity header
  (token passthrough) only if `ai-agent` is exposed as a standalone API to third parties.
- **True token streaming** — stream live vLLM tokens (instead of pseudo-streaming) and emit the
  citation block as the final chunks; a real `ai-agent` change touching `run_agent` and the §7
  contract.

---

## Open decisions — resolved

| Decision | Resolution |
|---|---|
| Conversation titles | First-message snippet for v1 (free). |
| Password reset | Admin-initiated only (no email; air-gap friendly). |
| Session store | Server-side, revocable sessions (not JWT). |
| Existing Open-WebUI users | Fresh start — clean break, different auth model. |
| OIDC group→role mapping | Configurable `group_roles` map; **default-deny to `user`** (a logged-in OIDC user with no mapped group gets the minimal `user` role: chat + own history, no admin). Admin only via the group map or admin promotion. |
| Rate-limiting | Login rate-limit in B2; per-user chat limits deferred to Effort C. |

---

## Dependencies

- **`rag_auth` / `chat-ui` backend:** Python 3.12, FastAPI, **PyJWT[crypto]** (JWT/JWKS),
  **httpx** (async discovery / token exchange), passlib/argon2-cffi (local password hashing),
  itsdangerous (signed cookie). No vendor IdP SDKs.
- **SPA:** vanilla JS + vendored MIT `marked` + `DOMPurify` (no build toolchain, no CDN —
  air-gap clean).
- **IdP:** any OIDC-compliant provider (Entra ID / Okta / Google / Keycloak / Auth0). On-prem AD
  via Entra/ADFS/Keycloak federation. Raw LDAP intentionally out of scope.
