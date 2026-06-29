# open-RAG-stack

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![Kubernetes 1.24+](https://img.shields.io/badge/Kubernetes-1.24%2B-blue)

A self-hosted Retrieval-Augmented Generation (RAG) stack running on Kubernetes. Ingest your own documents, query them through a custom AI agent backed by a local LLM, and chat through a built-in multi-user chat UI — entirely air-gapped if needed.

## Why this stack?

- **Fully self-hosted** — every component (LLM, embeddings, vector DB, agent, UI) runs in your cluster. No data leaves your network; it can run air-gapped.
- **Modular** — swap the LLM, embedding model, or web-search provider by editing `values.yaml` and one config block. Nothing is hardwired to a vendor.
- **Deploys anywhere** — plain Helm (`deploy/install.sh`) stands the whole stack up on any Kubernetes cluster. No GitOps controller required, though you can point one at the charts if you prefer.

### Cost comparison

Cloud AI APIs bill per token and per seat. A small team using document search and internal Q&A accumulates costs quickly:

| Capability | Hosted service | Typical cost | open-RAG-stack |
|---|---|---|---|
| LLM inference | OpenAI GPT-4o | ~$5–$15/1M tokens | Local vLLM — $0/token after hardware |
| Vector database | Pinecone Starter | ~$70/month | Qdrant self-hosted — free |
| Document ingestion | Azure AI Document Intelligence | ~$0.01/page | Self-hosted — free |
| Chat UI | ChatGPT Team | ~$25/user/month | Built-in chat UI — free, no per-seat license |
| Data privacy | Hosted APIs process your data on vendor infrastructure | Compliance risk | Air-gapped — your data never leaves |

**Hardware cost:** a single used RTX 3090 (24 GB VRAM) runs 7B–14B parameter models comfortably and is typically available for $600–$900. At $200–$500/month in avoided API costs, it pays for itself in a few months — and the inference cost per query is effectively zero thereafter.

> This stack is not a replacement for managed services in every scenario. If you need guaranteed SLAs, global scale, or have no GPU hardware, a managed API is likely the right call. But for a small-to-medium team processing internal documents on predictable workloads, self-hosted is often the better economic choice.

## Quick start at a glance

```bash
export NODE_IP=<your-gpu-node-ip>
./scripts/prereq-check.sh --gpu-node <your-node-hostname>  # validate cluster, GPU, storage, egress
./scripts/bootstrap.sh                                     # interactive: prompts for secrets, then deploys via Helm
# wait for pods — vLLM takes a few minutes to load the model (watch: kubectl get pods -A -w)
./scripts/link-scrape.sh https://docs.anthropic.com   # then answer the collection/vendor prompts
./scripts/rag-query.sh "What is Claude?"
```

See [Quick start](#quick-start) below for the full, step-by-step version (model selection, web search, storage).

## Architecture

![open-RAG-stack architecture](docs/rag-architecture.drawio.png)

<details>
<summary>Text version of the diagram</summary>

```
Query & response path
  User → chat-ui (:30086) → ai-agent (:30081)
  ai-agent runs hybrid retrieval:
    • embed query         → embedding service   (:30082, nomic-embed-text-v1.5)
    • vector search       → Qdrant              (:30333)
    • lexical search BM25 → ingestion SQLite FTS5 (:30083)
    • fuse (RRF) + rerank → reranker            (:30084, bge-reranker-v2-m3) → top-5
    • generate answer     → vLLM                (:30000, local LLM)
  ai-agent → chat-ui: answer + sources + verified citations
  optional: web_search   → SearXNG             (:8080, off by default)

Ingestion pipeline
  Sources: Watch folder (drop) · Web URLs (/ingest, /ingest/deep) · RAG Admin UI (:8005)
    → ingestion service (:30083): PyMuPDF + Tesseract OCR, recursive chunk 256/64, page-aware
        → embedding service → Qdrant      (vector index)
        → SQLite FTS5                     (BM25 lexical index)

Logical flow is identical for Docker Compose and Kubernetes; only the physical
layout differs (GPU sharing, host ports vs NodePorts). See the diagram's
"Deployment notes" box.
```

</details>

### Request flow

1. User sends a message in the **chat UI**
2. chat-ui forwards it to **ai-agent** via the OpenAI-compatible `/v1/chat/completions` endpoint
3. ai-agent decides which tool to call:
   - `rag_search` — hybrid retrieval: embeds the query and runs vector search (Qdrant) + lexical BM25 search (SQLite FTS5) across all collections, fuses the two with RRF into a top-20 pool, then reranks via cross-encoder to top-5 (optional query rewriting is off by default — see `docs/ENHANCEMENT-PLAN.md`)
   - `web_search` — calls your configured web search provider (Brave, SearXNG, Serper, or Tavily)
4. Tool results are injected back into the LLM context
5. **vllm-server** generates the final response
6. ai-agent returns the answer plus a `sources` list to chat-ui, which streams it to the browser with inline page-level citations

### Ingestion flow

```
URL or file  →  ingestion service  →  embedding service  →  Qdrant
               (scrape + chunk)       (nomic-embed-text)   (vector store)
```

Use `scripts/link-scrape.sh` to ingest a URL, or POST directly to the ingestion API.

---

## Services

| Service | Image | Port | Description |
|---|---|---|---|
| `chat-ui` | `ghcr.io/benwold-lgtm/open-rag-chat-ui` | 30086 | Chat UI — local + OIDC sign-in, per-user conversations, streaming answers with citations |
| `ai-agent` | `ghcr.io/benwold-lgtm/open-rag-ai-agent` | 30081 | RAG agent — calls vLLM + Qdrant |
| `embedding` | `ghcr.io/benwold-lgtm/open-rag-embedding` | 30082 | Embedding service (nomic-embed-text-v1.5) |
| `ingestion` | `ghcr.io/benwold-lgtm/open-rag-ingestion` | 30083 | Document ingestion pipeline |
| `reranker` | `ghcr.io/benwold-lgtm/open-rag-reranker` | 30084 | Cross-encoder reranker (BAAI/bge-reranker-v2-m3) |
| `rag-admin` | `ghcr.io/benwold-lgtm/open-rag-admin` | 30085 | Admin UI — document ingestion/management (proxies to ingestion) |
| `vllm-server` | `vllm/vllm-openai` | 30000 | LLM inference (OpenAI-compatible) |
| `qdrant` | `qdrant/qdrant` | 30333 | Vector database |

---

## Prerequisites

- Kubernetes cluster (tested on bare-metal k8s — kubeadm, k3s, or similar; k3s includes `local-path` out of the box)
- A GPU node with NVIDIA drivers and the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) installed
- Sufficient local disk on the GPU node for model weights (10–30 GB per model; 500 Gi allocated by default)
- `kubectl` and `helm` configured on your admin host
- A [Hugging Face](https://huggingface.co/) account and access token for model download

### Resource expectations

These are the defaults baked into the charts — tune them in each `values.yaml` for your hardware.

| Component | GPU | CPU (req/limit) | RAM (req/limit) | Storage |
|---|---|---|---|---|
| vLLM server | 1 × 24 GB GPU (RTX 3090/4090-class); `0.90` GPU-mem util | 4 / 8 | 16 Gi / 32 Gi (+16 Gi shared mem) | 500 Gi local-path (model weights) |
| embedding | optional (uses GPU if present, else CPU) | 2 / 8 | 4 Gi / 24 Gi | — |
| qdrant | — | 0.25 / 1 | 0.5 Gi / 2 Gi | 50 Gi local-path |
| ingestion | — | 0.5 / 2 | 1 Gi / 6 Gi | 5 Gi local-path |
| chat-ui | — | 0.1 / 1 | 128 Mi / 512 Mi | 2 Gi local-path |
| ai-agent | — | 0.25 / 0.5 | 0.25 Gi / 0.5 Gi | — |

In practice: a single GPU node with a **24 GB card**, ~**32–48 GB system RAM**, and ~**560 Gi free local disk** runs the whole AI stack. All persistent storage uses the `local-path` provisioner — no NFS or external storage required. The control-plane/worker VMs that host Qdrant, ingestion, and the chat UI are lightweight by comparison.

---

## Before you deploy

A few decisions to make up front — each maps to a value you'll set during configuration:

- **LLM + where it runs** — by default, vLLM serves a Hugging Face model locally on your GPU. Alternatively, point `ai-agent` at any OpenAI-compatible endpoint (another host or a hosted API) via `VLLM_BASE_URL`. Either way, decide the model ID.
- **Embedding model** — defaults to `nomic-embed-text-v1.5` (768 dims). If you change it, update `ingestion` `embeddingDim` to match.
- **Web search provider** — pick one (Brave, SearXNG, Serper, Tavily) or none. All but SearXNG need an API key; RAG search over your own documents works without any.
- **Vector DB API key** — you choose the `QDRANT_API_KEY` value at bootstrap (any string).
- **Storage** — local disk on the GPU node. All charts use the `local-path` StorageClass (installed by bootstrap if not present). No NFS required.
- **GPU node + access IP** — the node hostname for `nodeSelector`, and its IP (`NODE_IP`) for NodePort access.
- **Container images** — use the project's published `open-rag-*` images, or fork and let CI build your own.

---

## Quick start

### 1. Configure values

Edit each chart's `values.yaml` before deploying. At minimum:

**`ai-stack/charts/vllm-server/values.yaml`**
```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"   # or any HF model on Hugging Face

nodeSelector:
  kubernetes.io/hostname: your-gpu-node
```

**`ai-stack/charts/ai-agent/values.yaml`**
```yaml
vllm:
  baseUrl: "http://<gpu-node-ip>:30000/v1"
  model: "mistralai/Mistral-7B-Instruct-v0.3"

nodeSelector:
  kubernetes.io/hostname: your-gpu-node
```

Repeat `nodeSelector` for `embedding`, `ingestion`, `qdrant`, and `chat-ui` charts.

### 2. Configure web search (optional)

Web search is **off by default** (`provider: none`) — RAG over your own documents needs no provider. To enable it, just set the provider; **no code changes are required**:

- **Helm** — set `webSearch.provider` in `ai-stack/charts/ai-agent/values.yaml` to one of the options below, and add the matching API key to the `ai-agent-secrets` secret.
- **Docker Compose** — set `WEB_SEARCH_PROVIDER` (and the key) in the `ai-agent` service environment.

Options (the provider is read from the `WEB_SEARCH_PROVIDER` env var at runtime):

- **SearXNG** — self-hosted, no API key required; set `SEARXNG_URL` (deploy `ai-stack/charts/searxng` first)
- **Brave Search** — `BRAVE_API_KEY`
- **Serper** — `SERPER_API_KEY`
- **Tavily** — `TAVILY_API_KEY`

### 3. Bootstrap the cluster

```bash
# Set your environment
export NODE_IP=192.168.1.x          # GPU node IP

# Run bootstrap: installs local-path-provisioner, creates namespaces + secrets,
# then deploys every service with Helm.
./scripts/bootstrap.sh
```

The bootstrap script will interactively prompt for:
- `WEB_SEARCH_API_KEY` — leave blank if using SearXNG
- `QDRANT_API_KEY` — any string you choose
- `HF_TOKEN` — your Hugging Face access token
- `ghcr-pull-secret` (optional) — only if your GHCR images are private

### 3b. Qwen3 advanced mode (optional)

The default vLLM deployment runs any standard HuggingFace model cleanly. If you are running **Qwen3 on a single RTX 3090** and want to enable MTP speculative decoding + Genesis memory patches, flip the opt-in flag in `ai-stack/charts/vllm-server/values.yaml`:

```yaml
patches:
  enabled: true
```

This adds an `initContainer` that clones the Genesis and recipe patch repos, pins to a tested nightly image digest, and applies Qwen3-specific flags (`--reasoning-parser qwen3`, `--quantization auto_round`, `--kv-cache-dtype fp8_e5m2`, etc.). Leave `enabled: false` for all other models.

### 4. Deploy the services

`bootstrap.sh` deploys everything for you on its final step. If you change a chart or a `values.yaml` later, re-deploy with the same idempotent Helm script:

```bash
./deploy/install.sh
```

This runs `helm upgrade --install` for each service into its namespace — no GitOps controller required, so it works on any Kubernetes cluster.

**Using a GitOps tool instead?** The Helm charts under `ai-stack/charts/<service>/` are the deployable unit. Point ArgoCD, Flux, or Rancher Fleet at them and let your controller manage the rollout — you don't need `deploy/install.sh` in that case.

### 5. Verify everything is running

Once pods are `Running` (`kubectl get pods -A`), confirm each service responds. Set `NODE_IP` to your GPU node's IP:

```bash
export NODE_IP=<your-gpu-node-ip>

curl -s -o /dev/null -w "chat-ui     %{http_code}\n"  http://$NODE_IP:30086
curl -s -o /dev/null -w "vllm        %{http_code}\n"  http://$NODE_IP:30000/health
curl -s -o /dev/null -w "embedding   %{http_code}\n"  http://$NODE_IP:30082/health
curl -s -o /dev/null -w "ingestion   %{http_code}\n"  http://$NODE_IP:30083/health
curl -s -o /dev/null -w "ai-agent    %{http_code}\n"  http://$NODE_IP:30081/health
curl -s -o /dev/null -w "qdrant      %{http_code}\n"  http://$NODE_IP:30333/
```

Every service should return `200`. If a service returns `000` or nothing, check its pod logs with the matching script in `scripts/` (e.g. `./scripts/vllm-log.sh`).

---

## Forking this repo

If you fork this repo and want the CI to build and push images to your own GHCR namespace:

1. Fork to your GitHub account.
2. Push a commit to `main` — the `build-*.yml` workflows authenticate with the built-in `GITHUB_TOKEN` (no secret setup required) and push the custom images to `ghcr.io/<your-username>/open-rag-*` (ai-agent, embedding, ingestion, chat-ui, rag-admin, reranker).
3. Update `image.repository` in each chart's `values.yaml` to point to your GHCR namespace.

Each build also runs a **Trivy** vulnerability scan (results appear in the repo's **Security → Code scanning** tab) and publishes a **CycloneDX SBOM** as a workflow artifact. The scan is report-only — it does not fail the build on base-image CVEs.

**Making packages public (optional but simpler):** After the first CI run, go to each package on your GitHub profile → Change visibility → Public. This allows Kubernetes to pull without a pull secret.

---

## Ingesting documents

Documents are grouped into **collections** and tagged with a **vendor**. The agent searches across all collections at query time. There are four ways to get content in.

> **If you set a `SERVICE_TOKEN`** (see [Hardening → Authentication](#authentication)), the ingestion endpoints require it. The `scripts/*.sh` helpers send it automatically once you `export SERVICE_TOKEN=<value>`; for raw `curl` calls below, add `-H "Authorization: Bearer $SERVICE_TOKEN"`. With the data plane left open (no token), the calls work as written.

### 1. Scrape a single page or crawl a whole site

```bash
# Ingest a single URL
./scripts/link-scrape.sh https://docs.example.com/page

# Deep crawl — follows links to scrape an entire site
# (prompts for max depth, max pages, and an optional URL-pattern filter)
./scripts/link-scrape.sh https://docs.example.com --deep
```

#### Deep crawl explained

Deep crawl turns **one starting URL into many documents**. Beginning at the page you give it, the crawler follows the links on that page — and the links on *those* pages — ingesting each one. It's the fastest way to pull an entire documentation site, knowledge base, or product section into a collection in a single step. You stay in control with three settings:

| Setting | What it does | Guidance |
|---|---|---|
| **Max depth** | How many link-hops to follow from the starting page. Depth `1` ingests only pages linked **directly** from your URL; depth `2` also follows the links on those pages; and so on. | Higher depth = broader coverage, but slower and more likely to pull in unrelated pages. Start at `2`. Range 1–5. |
| **Max pages** | A hard cap on the **total** number of pages fetched, regardless of depth — a safety brake so a crawl of a large site can't balloon. | Set it a little above what you expect to need. Start at `30`. Range 1–200. |
| **URL pattern filter** *(optional)* | Restricts the crawl to URLs matching a wildcard pattern, so you ingest only the relevant part of a site. `*` matches any characters. | Leave blank to follow every link (within the depth/page limits). See the examples below. |

**URL pattern filter — examples.** Suppose you start at `https://example.com/docs/intro` and want only the documentation, not the blog or marketing pages:

| Pattern | Crawls | Skips |
|---|---|---|
| `*/docs/*` | `…/docs/install`, `…/docs/api/auth` | `…/blog/news`, `…/pricing` |
| `*example.com*` | any page on the `example.com` site | links that lead off to other domains |
| *(blank)* | every linked page, any URL | nothing — bounded only by depth & max pages |

**Tips for first-time users**
- **Start small.** The defaults (`2` / `30`, with a tight pattern) are deliberately conservative. It's faster to re-run a crawl wider than to clean up an over-broad one.
- **Use the filter to stay on-topic.** A pattern keeps the crawl from wandering into login pages, changelog archives, or other domains — which keeps your collection focused and your search results relevant.
- **Each page is its own document.** Crawled pages appear individually in the admin UI's document list, so you can spot and delete any strays afterward.

> Also available in the **RAG Admin UI** (`:30085`, or `:8005` under Docker Compose) at *Add Content → Deep crawl options*, where each field shows this guidance inline.

### 2. Scrape many URLs at once (batch)

POST a list of URLs to `/ingest/batch`. Each entry sets its own collection and vendor, so one call can populate multiple collections:

```bash
curl -X POST http://<your-gpu-node-ip>:30083/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"url": "https://blogs.nvidia.com/blog/enterprise-reference-architectures/", "collection": "nvidia-ai-factory", "vendor": "NVIDIA"},
      {"url": "https://docs.nvidia.com/enterprise-reference-architectures/index.html", "collection": "nvidia-ai-factory", "vendor": "NVIDIA"},
      {"url": "https://www.dell.com/en-us/blog/how-dell-makes-the-ai-factory-real/", "collection": "dell-ai-factory", "vendor": "Dell"},
      {"url": "https://www.dell.com/en-us/blog/securing-the-ai-factory/", "collection": "dell-ai-factory", "vendor": "Dell"}
    ]
  }'
```

`access_roles` (default `["all"]`) and `classification` (default `"public"`) can be set per document but are optional.

### 3. Upload a file (PDF, txt, or md)

```bash
curl -X POST http://<your-gpu-node-ip>:30083/ingest/document \
  -F "file=@./whitepaper.pdf" \
  -F "collection=nvidia-ai-factory" \
  -F "vendor=NVIDIA"
```

### 4. Drop files into a watch folder

Enable `watchDir` in `ai-stack/charts/ingestion/values.yaml` and point `hostPath` at a directory on the node. Files placed in `<watchDir>/<vendor>/` are ingested automatically — the subfolder name becomes both the vendor tag and the collection. Processed files move to a `processed/` subfolder.

### Interactive API (no UI required)

The ingestion service is a FastAPI app, so it serves a browser-based Swagger UI at:

```
http://<your-gpu-node-ip>:30083/docs
```

From there you can fill in and POST to `/ingest/url`, `/ingest/batch`, `/ingest/deep`, and `/ingest/document` without writing curl by hand. (The chat UI is the front-end for *querying* — it does not handle ingestion.)

### Check status and query

```bash
# List recent ingestion jobs (or pass a doc_id for detail)
./scripts/ingest-status.sh

# Query the agent from the CLI
./scripts/rag-query.sh "What is an AI factory reference architecture?"
```

---

## Repository structure

```
open-RAG-stack/
├── .github/workflows/        # CI: build & push Docker images to ghcr.io
├── ai-stack/
│   ├── charts/               # Helm charts — one per service
│   │   ├── ai-agent/
│   │   ├── chat-ui/
│   │   ├── embedding/
│   │   ├── ingestion/
│   │   ├── qdrant/
│   │   └── vllm-server/
│   ├── lib/
│   │   └── rag_auth/         # shared auth module (scopes, sessions, OIDC) — bundled into chat-ui
│   └── services/             # Python microservice source
│       ├── ai-agent/         # RAG agent (FastAPI)
│       ├── chat-ui/          # Chat UI — FastAPI API + vanilla-JS SPA
│       ├── embedding/        # Embedding service (FastAPI)
│       └── ingestion/        # Ingestion pipeline (FastAPI)
├── deploy/                   # install.sh — Helm deploy/upgrade for all services
├── docs/                     # Architecture diagram (.drawio source + .png)
├── scripts/                  # Helper scripts — see Scripts reference below
└── LICENSE                   # MIT
```

---

## Scripts reference

All scripts live in `scripts/` (except `install.sh`, which is in `deploy/`). The ingest/query/log scripts honour `NODE_IP`, `INGESTION_URL`, and `AI_AGENT_URL` environment variables so you can point them at your cluster without editing files.

| Script | Purpose |
|---|---|
| `scripts/prereq-check.sh` | Validate prerequisites before bootstrap (cluster, GPU, storage, egress) |
| `scripts/bootstrap.sh` | One-time setup: storage classes, namespaces, secrets, then deploys the stack |
| `deploy/install.sh` | Deploy or upgrade all services via Helm (idempotent; re-run after editing charts/values) |
| `scripts/RAG-startup.sh` | Scale the whole stack back to 1 replica (resume after shutdown) |
| `scripts/RAG-shutdown.sh` | Scale the whole stack to 0 to free GPU and RAM (storage/secrets untouched) |
| `scripts/status.sh` | Pod overview across all stack namespaces |
| `scripts/link-scrape.sh` | Ingest a URL — single page or `--deep` site crawl |
| `scripts/ingest-status.sh` | List recent ingestion jobs, or inspect one by `doc_id` |
| `scripts/rag-query.sh` | Send a question to the agent from the CLI |
| `scripts/ai-agent-log.sh` | Tail ai-agent logs |
| `scripts/embedding-log.sh` | Tail embedding logs |
| `scripts/ingestion-log.sh` | Tail ingestion logs |
| `scripts/qdrant-log.sh` | Tail Qdrant logs |
| `scripts/vllm-log.sh` | Tail vLLM server logs |

---

## Hardening for production use

The defaults in this repo are designed for getting started quickly. Before putting this stack in front of employees or on a network that isn't fully trusted, consider the following.

### Authentication

The **chat UI** authenticates every request. The first account to register becomes the admin; subsequent local accounts stay pending until an admin approves them. It also supports **OIDC single sign-on** (Entra, Okta, Google, Keycloak — configure via the chart `oidc.*` values or the `.env` OIDC block) and a **break-glass** recovery admin independent of the user table and the IdP. In production (`ENVIRONMENT=production`) it **fails closed** — it refuses to boot without a `SESSION_SECRET` and at least one auth method. On any untrusted network, set a strong `SESSION_SECRET`, serve over TLS, and set `cookieSecure: true`.

The **RAG Admin UI** (port 8005 / NodePort 30085) is unauthenticated by default. To require a login on every page and write/delete action, set `ADMIN_USER` and `ADMIN_PASSWORD` — in `.env` for Docker Compose, or as the `rag-admin-auth` secret with `auth.enabled: true` in the chart. The `/health` probe stays open. This gates the UI; keep network access LAN-scoped regardless.

The **data plane** — `ai-agent` (`/v1/chat/completions`, `/v1/models`) and `ingestion` (all data routes) — is protected by a shared machine-to-machine **`SERVICE_TOKEN`**. chat-ui, rag-admin and the helper scripts send it as `Authorization: Bearer $SERVICE_TOKEN`; ai-agent and ingestion require it on every non-health route. The Helm path **auto-generates one token** in `scripts/bootstrap.sh` and writes the same value into the `ai-agent-secrets`, `ingestion-secrets` and `chat-ui-secrets` secrets (add it to `rag-admin-secrets` yourself if you deploy rag-admin). For Docker Compose the data plane ships **open** (frictionless single-node testing); set `SERVICE_TOKEN` in `.env` (`openssl rand -hex 32`) to require it everywhere.

In Kubernetes (`ENVIRONMENT=production`) ai-agent and ingestion **fail closed** — they refuse to boot with no `SERVICE_TOKEN` configured. To run them open on a trusted, isolated LAN, set **`ALLOW_ANONYMOUS=true`** (`config.allowAnonymous: true` in their charts), which restores the legacy unauthenticated behavior. Two routes always stay open: each service's `/health` probe, and ingestion's page-image route (`/documents/{id}/pages/{page}/image`), which browsers load directly as `<img>` URLs and cannot send a bearer token — gate that one at the network layer.

Before exposing the **chat UI** beyond a trusted LAN:

- **Serve over TLS** (ingress or reverse proxy) and set `cookieSecure: true` (chart) / `COOKIE_SECURE=true` (Compose). Session cookies are `HttpOnly` + `SameSite=Lax`; the Secure flag is the missing piece on plain HTTP. It ships **off** so a bare NodePort works out of the box — turn it on once TLS is in front. The chat-ui chart ships an optional `ingress.yaml`: set `ingress.enabled: true` with your `host` and an `ingressClassName`, and either point `tls.secretName` at a TLS secret you create, or add a `cert-manager.io/cluster-issuer` annotation to have [cert-manager](https://cert-manager.io/) provision it. Example:

  ```yaml
  # chat-ui values.yaml
  config:
    cookieSecure: true
  ingress:
    enabled: true
    className: nginx
    host: chat.example.com
    annotations:
      cert-manager.io/cluster-issuer: letsencrypt-prod
    tls:
      enabled: true
      secretName: chat-ui-tls
  ```
- **Set a strong, persistent `SESSION_SECRET`** (the Helm path auto-generates one into `chat-ui-secrets`; for Compose, generate with `openssl rand -hex 32`). Rotating it invalidates all sessions.
- **Tighten registration** for a closed user base: leave `REQUIRE_APPROVAL=true` (the default — new accounts wait for an admin), or set `REGISTRATION_ENABLED=false` and create accounts as an admin. With OIDC, set `OIDC_DEFAULT_ROLE=""` to deny users who match no mapped group.
- The login **rate-limiter is per-process**, which is exact for the default single replica. If you scale chat-ui to multiple replicas, put a shared limiter (e.g. a reverse-proxy or Redis-backed limit) in front — the in-memory counter won't be shared across pods. Responses also carry `Content-Security-Policy`, `X-Frame-Options: DENY`, and `X-Content-Type-Options: nosniff`.

### Network isolation (NetworkPolicies)

By default, every pod can reach every other pod in the cluster. The `qdrant`, `ai-agent` and `ingestion` charts ship a `templates/networkpolicy.yaml` that default-denies ingress and allows only the known in-cluster callers (qdrant ⇐ ai-agent + ingestion; ai-agent ⇐ chat-ui; ingestion ⇐ ai-agent + rag-admin). It is **off by default** so a first deploy can't be broken by a CNI you didn't expect. Turn it on per chart:

```yaml
# values.yaml (qdrant / ai-agent / ingestion)
networkPolicy:
  enabled: true
```

**CNI caveat:** NetworkPolicy is only enforced by a CNI that implements it — **Calico** and **Cilium** do; **flannel does not** (it silently ignores the policy, so you get no isolation). Confirm your CNI before relying on this.

**NodePort caveat:** enabling the policy also blocks the NodePort path for the gated services. Inline page-image citations (the browser loads them from ingestion's `:30083`) and the `scripts/*.sh` helpers (which hit the NodePorts from outside the cluster) will stop working until you add an `allow-from-*` rule for your ingress/source. For other namespaces (`embedding`, `ai-stack`, `chat-ui`), apply the same default-deny + `allow-from-*` pattern by hand; refer to the [Architecture](#architecture) diagram for the exact call paths.

### Pod security

The first-party services (`chat-ui`, `ai-agent`, `ingestion`, `rag-admin`) and `qdrant` already run non-root with `allowPrivilegeEscalation: false`:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
```

`vllm-server` stays **root** on purpose — it's the upstream GPU image and needs device access; it now ships `/health` liveness/readiness probes so a wedged model load is restarted. If you add more services, apply the same non-root `securityContext` to their `spec.template.spec`.

### Secrets at rest

The bootstrap script creates plain Kubernetes Secrets (base64-encoded, not encrypted). If your cluster does not have [encryption at rest](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/) enabled, consider using an external secrets manager (Vault, AWS Secrets Manager, Azure Key Vault) via the [External Secrets Operator](https://external-secrets.io/).

---

## Backup & restore

The stack keeps state in three places: **Qdrant** (the vectors), the **chat-ui SQLite DB** (users, conversations, messages), and the **ingestion SQLite DB + files** (the FTS index, document metadata, and uploaded files / page images). Two scripts back these up and restore them for the **Docker Compose** deployment:

```bash
./scripts/backup.sh                 # -> ./backups/<UTC timestamp>/
./scripts/backup.sh /mnt/backups    # or a directory you choose
./scripts/restore.sh ./backups/20260629-220900Z
```

- **Backup is online and consistent** — it takes a Qdrant snapshot per collection and a SQLite `.backup` of each DB, so you don't need to stop the stack. Run it from the repo root; schedule it with cron for regular backups.
- **Restore overwrites current data.** It recovers each Qdrant collection from its snapshot and briefly stops chat-ui / ingestion to swap their DB files back in. Take a fresh backup first if unsure.
- **Secrets are not included.** Back up your `.env` (Compose) or Kubernetes Secrets separately and securely — they hold `SERVICE_TOKEN`, `SESSION_SECRET`, API keys, etc.
- If Qdrant has an API key set, export `QDRANT_API_KEY` (and `QDRANT_URL` if not `http://localhost:6333`) before running either script.

For a **Kubernetes** deployment, the same primitives apply but the transport differs: use the Qdrant snapshot API (via a port-forward or the NodePort), and `kubectl exec` the SQLite `.backup` out of each pod — or snapshot the PVCs at the storage layer.

### Known limitation: single-writer / single-replica

Both SQLite databases and the default single-replica Qdrant are **single-writer**. Run **one replica** of chat-ui and ingestion — scaling them out would corrupt the SQLite files, and there is no built-in HA / clustering story. This is fine for the target single-node deployment; if you need horizontal scale or high availability, you'd migrate the SQLite stores to a networked database (e.g. Postgres) and run Qdrant in its clustered mode. Regular backups (above) are the recommended safety net.

---

## Observability (metrics)

The four first-party services — **chat-ui, ai-agent, ingestion, rag-admin** — each expose Prometheus metrics at **`GET /metrics`** (request rate, latency histograms, in-progress requests, status classes). Two of the infrastructure components already ship their own metrics: **vLLM** (`/metrics` on its API port) and **Qdrant** (`/metrics` on `:6333`).

`/metrics` is **unauthenticated**, exactly like `/health` — it exposes only counts and latencies, no secrets. Keep it network-gated the same way as the rest of the data plane (LAN-only / NetworkPolicy).

**Docker Compose** — point Prometheus at the services. If your Prometheus shares the `rag-net` network, scrape by service name; otherwise use the host-mapped ports:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: open-rag-stack
    metrics_path: /metrics
    static_configs:
      # on the rag-net network (container name : container port):
      - targets: ["ai-agent:8000", "ingestion:8002", "chat-ui:8006", "rag-admin:8005"]
      # …or from outside the compose network, via the host-mapped ports:
      # - targets: ["<host-ip>:8004", "<host-ip>:8002", "<host-ip>:3001", "<host-ip>:8005"]
```

**Kubernetes** — the four charts set `prometheus.io/scrape: "true"` + `prometheus.io/path` + `prometheus.io/port` pod annotations, so a Prometheus configured for annotation-based pod discovery picks them up automatically. (No `ServiceMonitor` is shipped — that assumes the Prometheus Operator, which this stack doesn't bundle.)

A minimal starter dashboard: request rate (`rate(http_requests_total[5m])`), p95 latency (`histogram_quantile(0.95, …)`), and error ratio (4xx/5xx over total). GPU/utilisation metrics come from DCGM-exporter and vLLM separately.

---

## Troubleshooting

**vLLM pod stays in `Pending`**
- Check that the NVIDIA device plugin is installed: `kubectl get pods -n kube-system | grep nvidia`
- Verify the node has a GPU resource: `kubectl describe node <your-gpu-node> | grep nvidia.com/gpu`
- Confirm `nodeSelector` in `vllm-server/values.yaml` matches the node's hostname exactly.

**vLLM pod crashes on startup**
- Check logs: `kubectl logs -n ai-stack deploy/vllm-server --previous`
- If you have `patches.enabled: true`, confirm the patch repos are reachable from your cluster (requires internet egress).
- For a standard model (Mistral, Llama, etc.) leave `patches.enabled: false` — the clean startup path is the default.

**PVC stays in `Pending`**
- Confirm `local-path` StorageClass is installed: `kubectl get storageclass local-path`
- If missing, run: `kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml`
- Check provisioner logs: `kubectl logs -n local-path-storage -l app=local-path-provisioner`
- Confirm the node has enough free disk space: `df -h` on the GPU node. The vLLM PVC requests 500 Gi by default.

**Images fail to pull (`ErrImagePull` / `ImagePullBackOff`)**
- GHCR packages are private by default. Either make them public (see [Forking this repo](#forking-this-repo)) or run bootstrap.sh and choose `y` when prompted to create `ghcr-pull-secret`.

**Qdrant pod has no API key / RAG queries return 401**
- Confirm `qdrant-secrets` exists in the `qdrant` namespace: `kubectl get secret qdrant-secrets -n qdrant`
- If missing, run bootstrap.sh again — it skips existing secrets so re-running is safe.

**401 from ingestion or ai-agent (or chat answers / admin actions fail with 401)**
- The data plane is token-protected. Confirm the same `SERVICE_TOKEN` is in every relevant secret: `kubectl get secret ai-agent-secrets -n ai-agent ingestion-secrets -n ingestion chat-ui-secrets -n chat-ui -o jsonpath='{..SERVICE_TOKEN}'` should print the **same** base64 value three times. Re-run `bootstrap.sh` (it reuses the existing token) if one is missing.
- For the helper scripts, export the token first: `export SERVICE_TOKEN=<value>` before running `rag-query.sh` / `ingest-status.sh` / `link-scrape.sh`.
- `ai-agent`/`ingestion` **refuse to boot** in `ENVIRONMENT=production` with no token: check their logs for the fail-closed warning, set the token, or set `ALLOW_ANONYMOUS=true` for a trusted LAN.

**Chat UI answers fail or hang**
- Confirm ai-agent is running: `kubectl get pods -n ai-agent`
- Check `aiAgent.url` in `chat-ui/values.yaml` (or `AI_AGENT_URL` in Compose) points to ai-agent's cluster DNS / service name, not directly to vLLM.
- If logins don't stick, you're likely on plain HTTP with `cookieSecure: true` — set it to `false` for bare NodePort/LAN access, or put the UI behind TLS.
- In production the pod refuses to boot without a `SESSION_SECRET`; check the `chat-ui-secrets` secret exists in the `chat-ui` namespace.

### Common gotchas

- **Embedding dimension must match the model.** `ingestion` `embeddingDim` defaults to `768` for `nomic-embed-text-v1.5`. If you swap the embedding model, update `embeddingDim` to the new model's dimension *and re-ingest* — a mismatch makes every search silently return nothing.
- **NodePorts must be unique.** Check the [NodePort table](#services) before adding a service; a duplicate port makes the new Service fail to create.
- **vLLM flags are model-specific.** The default deployment is clean — no extra parsers or quantization flags. If you set `patches.enabled: true`, those settings are Qwen3-specific and will break tool calling on other models.
- **Collection and vendor names are case-sensitive.** `Cisco` and `cisco` are different collections; keep them consistent between ingestion and querying.
- **`pullPolicy: Always` + private images.** If your `open-rag-*` packages are private, pods will `ImagePullBackOff` until you add `ghcr-pull-secret` or make the packages public.

---

## Secrets reference

All secrets are created by `scripts/bootstrap.sh` and stored in Kubernetes — nothing is committed to the repo.

| Secret name | Namespace | Keys | Purpose |
|---|---|---|---|
| `ai-agent-secrets` | `ai-agent` | `BRAVE_API_KEY`, `QDRANT_API_KEY`, `SERVICE_TOKEN` | Web search + Qdrant auth + data-plane token for ai-agent |
| `ingestion-secrets` | `ingestion` | `SERVICE_TOKEN` | Data-plane token ingestion verifies |
| `chat-ui-secrets` | `chat-ui` | `SESSION_SECRET`, `SERVICE_TOKEN` | Session signing + data-plane token chat-ui sends |
| `qdrant-secrets` | `qdrant` | `QDRANT_API_KEY` | Qdrant API key |
| `hf-token-secret` | `ai-stack` | `token` | Hugging Face token for vLLM model download |
| `ghcr-pull-secret` | `ingestion` | Docker config | Pull custom images from ghcr.io |

`SERVICE_TOKEN` is one shared value — `bootstrap.sh` generates it once and writes the same string into all three secrets above. If you deploy `rag-admin`, create `rag-admin-secrets` in its namespace with the same `SERVICE_TOKEN` so its proxied calls to ingestion are authorized.

---

## Contributing & support

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and tests, and report security issues privately per [SECURITY.md](SECURITY.md).

This is a personal project shared as-is. It's maintained on a **best-effort basis with no SLA or commercial support**, and comes with no warranty (see [LICENSE](LICENSE)). If you run it in production, plan to maintain your own fork.
