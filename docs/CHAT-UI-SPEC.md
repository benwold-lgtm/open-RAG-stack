# Chat UI — Design Spec (Open-WebUI Replacement)

**Status:** Draft for review (spec-first; no code yet)
**Last Updated:** 2026-06-28
**Decision:** Build a first-party, multi-user chat UI to replace Open-WebUI.
**Auth model (refined 2026-06-28):** OIDC SSO (Entra/Okta/Google/Keycloak) **+ local-user fallback**. See §5.

---

## Review focus — three things to look at first

These are the parts most worth your attention on review; the rest is well-trodden:

1. **Streaming ↔ citations contract (§7).** The one real technical subtlety. The agent
   appends the Sources / Verified-quotes blocks *after* generation; with token streaming
   those must arrive after the model tokens, and `ai-agent` likely needs SSE streaming added.
   Scoped and small, but design it deliberately.
2. **Auth is the bulk of the work (§5, milestone M1).** "Full multi-user" — now with OIDC SSO
   **plus** a local-account fallback — roughly doubles the timeline vs a single-admin MVP, but
   it's the right call for a resell-grade product. Spec-first surfaces that now, not mid-build.
3. **Open decisions (§13).** A few small calls (conversation titles, password reset, sessions
   vs JWT, OIDC group→role mapping) with leanings noted; settling them unblocks M1.

---

## 1. Why

Open-WebUI is the only third-party component in the stack with a **branding/redistribution
license constraint** (its license restricts altering the "Open WebUI" branding without an
enterprise license). That conflicts with the project's core goal: **anyone can download,
run, and rebrand the stack for free, with no fee to any vendor.**

Replacing it with a component we own removes that constraint entirely and makes full
white-labeling (name, logo, colors) a first-class, config-driven feature — important for
both end-users and resellers.

**Non-goal:** re-implementing Open-WebUI. We only need the slice this stack actually uses.

---

## 2. Scope

**In scope**
- Real-time chat against the existing `ai-agent` OpenAI-compatible API (with streaming).
- Markdown rendering of assistant output, including the agent's **Sources**, inline
  **Referenced pages** images, and **Verified quotes** (Phases 4d/4e).
- Multi-conversation history, persisted per user.
- **Multi-user auth**: accounts, registration, admin approval, roles. (Chosen model.)
- Model selection (from `/v1/models`).
- **Config-driven branding**: name, logo, colors, favicon per deployment.

**Out of scope** (Open-WebUI features this stack does not need)
- Document/RAG management — `rag-admin` owns ingestion; `ai-agent` owns retrieval.
- Pipelines, plugins/tools UI, multi-provider model management, image generation.

**Unchanged by this work**
- `ai-agent` (retrieval + citations), `ingestion`, `qdrant`, `rag-admin`.
  *Exception:* `ai-agent` likely needs **streaming support** added — see §7.

---

## 3. Architecture

A new service **`chat-ui`** = **FastAPI + a vanilla-JS single-page app + SQLite**, i.e. the
same shape as `rag-admin`. It takes Open-WebUI's slot (host `:3001` / NodePort `30080`) and
talks to the agent at `http://ai-agent:8000/v1`.

Rationale for vanilla-SPA over a React/Svelte build:
- **No build toolchain, no CDN** → stays air-gap-clean and trivially downloadable (a core
  project value). The team already maintains exactly this pattern in `rag-admin`.
- **Permissive deps only.** Two small MIT libraries, **vendored into the repo** (not from a
  CDN): a markdown renderer (`marked`) and an HTML sanitizer (`DOMPurify`). No copyleft, no
  commercial, no runtime third-party calls.

```
Browser ──(session cookie)──> chat-ui (FastAPI + SPA + SQLite: users, conversations, messages)
                                   │  proxies, persists, streams
                                   ▼
                              ai-agent  /v1/chat/completions  (+ /v1/models)
                                   │ (unchanged: RAG retrieval + citations)
                                   ▼
                     embedding · qdrant · ingestion · reranker · vLLM
```

The chat-ui backend is the **only** thing the browser talks to (it never sees the agent
directly), which gives us one clean place for auth, history, and CSRF.

---

## 4. Data model (SQLite, in the chat-ui service)

| Table | Columns (sketch) |
|---|---|
| `users` | `id, username, email, password_hash (nullable for SSO), auth_source('local'|'oidc'), oidc_sub, role('admin'|'user'), status('pending'|'active'|'disabled'), created_at` |
| `sessions` | `token, user_id, created_at, expires_at` (server-side, revocable) |
| `conversations` | `id, user_id, title, created_at, updated_at` |
| `messages` | `id, conversation_id, role('user'|'assistant'), content, created_at` |

Notes:
- Store the assistant's **raw markdown** (including the citation blocks) so a reload renders
  identically. Render to HTML client-side.
- Per-user isolation enforced in every query (`WHERE user_id = :me`).

---

## 5. Authentication (the largest piece)

**Design principle (project guidance): solve for ~90% of real deployments — two layers, not
five.**

- **Federated SSO via OIDC** — a single protocol that covers the common enterprise IdPs:
  **Microsoft Entra ID** (formerly Azure AD), **Okta**, **Google Workspace**, **Keycloak**,
  Auth0, etc. On-prem **Active Directory** is normally fronted by Entra/ADFS and speaks OIDC
  too, so OIDC reaches it without a separate integration. **Raw LDAP is intentionally out of
  scope** (rarely seen now) — revisit only if a specific deployment demands it.
- **Local accounts — always available, as the fallback.** Even when OIDC is configured, local
  login stays on so admins/users can still get in **when the IdP is unreachable** (provider
  outage, or an air-gapped site that can't reach a cloud IdP). Local accounts are also the
  default for small/self-hosted deployments that have no IdP at all. *This dual-path (federated
  primary + local fallback) is a hard requirement.*

**Local accounts**
- **Registration → `pending` → admin approval → `active`.** First registered user becomes
  `admin` + `active` automatically (bootstraps the system).
- Passwords hashed with bcrypt/argon2 (`passlib` / `argon2-cffi`); never stored plain.
- Password reset is **admin-initiated** (no email dependency — air-gap friendly).

**OIDC SSO (optional, config-driven)**
- Standard **Authorization Code + PKCE** flow against the org's IdP.
- **Auto-provision** a local `users` row on first successful SSO login; map IdP **group/role
  claims → app roles** (`admin`/`user`), with optional admin approval for first login.
- Config: `OIDC_ENABLED`, `OIDC_ISSUER`, `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`,
  `OIDC_REDIRECT_URL`, and a claim/group→role map. Disabled by default.
- Keep the **local-login route reachable even when OIDC is enabled** (a "sign in locally"
  link), so an IdP outage never locks everyone out.

**Sessions & roles (both paths converge here)**
- Server-side session token in `sessions`, delivered as a **signed, httpOnly, Secure,
  SameSite cookie**. Chosen over JWT because it's **revocable** (disable a user / force logout
  → kill sessions) and simpler. A federated login and a local login produce the *same* session.
- Roles: `admin` (manage users + settings) and `user`. Admin panel to approve / disable /
  promote, regardless of how the account was created.
- **Config flags:** `REGISTRATION_ENABLED`, `REQUIRE_APPROVAL`, session TTL.

---

## 6. API (chat-ui backend → browser)

```
Auth     POST /api/auth/register | /api/auth/login | /api/auth/logout
         GET  /api/auth/me
Convos   GET  /api/conversations              POST /api/conversations
         PATCH /api/conversations/{id}  (rename)   DELETE /api/conversations/{id}
         GET  /api/conversations/{id}/messages
Chat     POST /api/conversations/{id}/chat   → streams; proxies to ai-agent, persists both turns
Models   GET  /api/models                    → proxies /v1/models
Admin    GET  /api/admin/users   POST /api/admin/users/{id}/{approve|disable|role}
Health   GET  /health  (unauthenticated, for probes)
```

All non-auth, non-health routes require a valid session; admin routes require `role=admin`.

---

## 7. Streaming + citations (the key technical subtlety)

- I verified the agent works with `stream:false`. For good UX it should support
  **`stream:true` (SSE)** — the RAG turn is ~5–9 s end-to-end, so token streaming massively
  improves *perceived* latency. **If the agent doesn't stream yet, adding it is a scoped
  `ai-agent` change**, tracked as part of this effort.
- **Subtlety:** the agent currently **appends** the Sources / Verified-quotes blocks *after*
  generation (post-processing). With streaming, those must arrive **after** the model tokens.
  Plan: stream the model's answer tokens, then emit the citation block as the final chunk(s).
  The chat-ui shows a "retrieving…" state before the first token (since retrieval precedes
  generation), then streams the answer, then renders citations/images. This needs a small,
  deliberate contract between agent and UI — call it out in the agent streaming work.

---

## 8. Branding (config-driven, the payoff)

Injected into the served SPA at render time (same technique `rag-admin` uses for its
auth note), driven per-deployment via env / Helm values:

| Var | Effect |
|---|---|
| `BRAND_NAME` | App title, login screen, header |
| `BRAND_LOGO_URL` | Header/login logo (served from the container or a URL) |
| `BRAND_PRIMARY_COLOR` | Accent color (single CSS variable) |
| `BRAND_FAVICON_URL` | Browser tab icon |

A reseller rebrands by setting four values — no fork, no code change.

---

## 9. Security checklist

- bcrypt/argon2 password hashing; login rate-limiting.
- httpOnly + Secure + SameSite cookies; **CSRF token** on state-changing requests.
- **Sanitize** all rendered markdown (DOMPurify) — LLM output is untrusted HTML.
- Allow inline images only from the ingestion public URL (the trust we already extend).
- Per-user data isolation on every query.
- No external network calls at runtime (air-gap parity with the rest of the stack).

---

## 10. UI structure (vanilla SPA)

- **Login / Register** views.
- **Main:** left sidebar (conversation list, "New chat", user menu → logout, admin link if
  admin); main pane (message thread, streaming composer with stop button, model dropdown);
  branded header.
- **Admin:** user table (approve / disable / set role).
- Light/dark theme (v-late, cheap once colors are CSS variables).

---

## 11. Repo changes

- `ai-stack/services/chat-ui/` — `main.py`, `requirements.txt`, `Dockerfile`, `static/` (vendored `marked`, `DOMPurify`).
- `ai-stack/charts/chat-ui/` — Deployment + Service (NodePort 30080), env/branding/auth-secret wiring (mirror existing charts).
- `.github/workflows/build-chat-ui.yml` — mirror an existing service build.
- `docker-compose.yml` — replace the `open-webui` service with `chat-ui`.
- `ai-agent` — add SSE streaming support (§7), if not already present.
- README, architecture diagram, services table — swap Open-WebUI → chat-ui; remove the
  Open-WebUI license note.
- Decommission Open-WebUI (image, env, volume) once chat-ui is validated.

---

## 12. Build milestones (internal sequencing; product-grade target)

1. **M1 — Auth core:** service skeleton, users/sessions; **local accounts** (register/login/logout, approval, roles) + admin user-management API; **OIDC SSO** (Entra/Okta/Google/Keycloak) with Auth-Code+PKCE, auto-provisioning, claim→role mapping, and the always-available local-login fallback.
2. **M2 — Conversations:** CRUD + per-user persistence.
3. **M3 — Chat:** proxy to agent, **streaming**, markdown + citation/image rendering (incl. the §7 contract).
4. **M4 — Branding + admin UI:** config-driven theme; user-management screen.
5. **M5 — Wiring:** chart, CI, compose swap, README/diagram; stand up alongside Open-WebUI.
6. **M6 — Hardening + cutover:** security pass, e2e test on the GPU node, decommission Open-WebUI.

**Rough effort:** ~2 weeks for product-grade multi-user.

---

## 13. Open decisions (resolve during review)

- **Conversation titles:** first-message snippet (free) vs a small LLM call to summarize (nicer, one extra call). *Lean: snippet for v1.*
- **Password reset:** admin-initiated only (no email in an air-gapped stack). *Confirm.*
- **Session store:** server-side sessions (recommended, revocable) vs JWT. *Lean: sessions.*
- **Existing Open-WebUI users:** migrate vs fresh start. *Lean: fresh — different auth model, clean break.*
- **OIDC group→role mapping:** trust an IdP group/role claim directly, vs default everyone to
  `user` and promote via an admin allow-list. *Lean: configurable claim/group→role map, default-deny to `user`.*
- **Rate-limiting / abuse:** scope for login; per-user chat rate limits later?

---

## 14. Reference: what we keep vs replace

| Open-WebUI gave us | Replacement |
|---|---|
| Chat + streaming | chat-ui SPA + agent SSE (§7) |
| Markdown + citation/image rendering | vendored `marked` + `DOMPurify` |
| Conversation history | SQLite (`conversations`, `messages`) |
| Multi-user auth | chat-ui auth (§5) — owned, no license constraint; **OIDC SSO + local fallback** |
| Model selection | proxy `/v1/models` |
| (RAG/docs, plugins, pipelines) | dropped — not used by this stack |
