# Open-RAG-Stack Enhancement Plan
## Goal: NVIDIA Blueprint Parity — Fully Internal Deployment

**Last Updated:** 2026-06-24
**Status:** In Progress
**Maintained by:** open-RAG-stack contributors

---

## Background

Open-RAG-Stack is a fully open-source RAG pipeline designed so anyone can run it for free (outside of a subscription LLM). Functionality was validated on a single RTX 3090 (24 GB VRAM). The production target for a medium business deployment is one or more **NVIDIA L40S GPUs** (48 GB VRAM each), which provide the headroom needed to run a 70B-class LLM alongside embedding and reranker services simultaneously.

The NVIDIA Enterprise RAG Blueprint (build.nvidia.com) was used as a reference for feature parity. That blueprint uses proprietary NIM microservices (Nemotron LLMs, nemotron-embed, nemotron-rerank), Elasticsearch as the default vector/search backend, and LangChain + LangGraph for orchestration. This plan achieves the same capabilities using open-source equivalents.

**Constraint:** The entire stack — at runtime — must make zero calls to external networks. No HuggingFace at runtime, no cloud search APIs, no external registries. Internet access is only permitted during the one-time setup/build phase.

---

## Architecture: Current vs Target

| Capability | Current | Target |
|---|---|---|
| Semantic search | ✅ nomic-embed-text-v1.5 | ✅ keep |
| Lexical / BM25 search | ❌ none | ✅ SQLite FTS5 (no new service) |
| Hybrid search fusion | ❌ none | ✅ RRF (Reciprocal Rank Fusion) |
| Reranker | ❌ none | ✅ BAAI/bge-reranker-v2-m3 |
| PDF parsing | ⚠️ flat text, no page metadata | ✅ PyMuPDF page-aware text + Tesseract OCR fallback for image-rich pages (see Phase 4c) |
| Chunking strategy | ⚠️ word-based | ✅ RecursiveCharacterTextSplitter |
| Multi-chunk per source | ❌ 1 chunk/URL max | ✅ 3 chunks/URL max |
| Query rewriting | ❌ none | ✅ uses existing LLM |
| Web search | ⚠️ Brave only (hardcoded) | ✅ SearXNG (self-hosted, default) or Brave/Serper/Tavily via env var |
| Model hosting | ⚠️ downloads at runtime | ✅ pre-cached to local/NFS path |
| Container images | ⚠️ pulls from GHCR/DockerHub | ✅ mirrored to internal registry |
| Multi-index search | ⚠️ all collections, no routing | ✅ keep (routing is future work) |

---

## Hardware Sizing Guide

This stack is hardware-agnostic — any CUDA-capable GPU node works. The table below covers two reference configurations.

| Configuration | GPU | VRAM | Suitable For |
|---|---|---|---|
| Development / testing | RTX 3090 | 24 GB | 7B–14B models; functional validation |
| Production (medium business) | NVIDIA L40S × 1 | 48 GB | 70B models + embedding + reranker on one card |
| Production (high throughput) | NVIDIA L40S × 2+ | 96 GB+ | 70B models with tensor parallelism |

**L40S single-card GPU budget (48 GB VRAM):**
- vLLM with Llama-3.3-70B-Instruct Q4: ~35 GB
- Embedding service (nomic-embed-text-v1.5): ~0.5 GB
- Reranker service (bge-reranker-v2-m3): ~1.5 GB
- **Total: ~37 GB — fits with headroom**

**RTX 3090 GPU budget (dev/test, 24 GB VRAM):**
- vLLM with 8B model Q4: ~8 GB
- Embedding: ~0.5 GB
- Reranker: ~1.5 GB (or CPU fallback to free VRAM for LLM)
- **Total: ~10 GB — fits for testing**

> To configure which node runs GPU workloads, set `nodeSelector` in each service's `values.yaml`.
> See `ai-stack/charts/*/values.yaml` — the field is `nodeSelector.kubernetes.io/hostname`.

---

## Implementation Phases

---

### Phase 1: Internal Hosting (Blockers)
> Must complete before any feature work. These are hidden external dependencies that break air-gap requirements.

| # | Task | Status | Notes |
|---|---|---|---|
| 1.1 | Pre-download all models to model storage path | ✅ Done | `scripts/download-models.sh` — set `MODEL_DIR` to your NFS mount or local path. LLM skipped pending model selection. |
| 1.2 | Add `HF_HOME` env var pointing to model storage in all service Helm charts | ✅ Done | `modelStorage` block added to `embedding` and `vllm-server` charts. Set `modelStorage.enabled: true` and `modelStorage.pvcName` in each chart's `values.yaml` to activate. |
| 1.3 | Replace Brave/Serper/Tavily with SearXNG | ✅ Done | `ai-agent/main.py` — `WEB_SEARCH_PROVIDER` env var dispatches to all 4 providers. Default: `none` (web search off). |
| 1.4 | Deploy SearXNG to K8s (new Helm chart) | ✅ Done | `ai-stack/charts/searxng/` — ClusterIP service, configmap-mounted settings.yml with JSON API enabled. Set `provider: searxng` in ai-agent values to activate. |
| 1.5 | Mirror container images to internal registry | ✅ Done | `scripts/mirror-images.sh` — pulls all 8 images, pushes to `$REGISTRY`, prints helm overrides. `ai-stack/registry-values.example.yaml` shows per-chart override values. `vllm-server` patches initContainer image moved from hardcoded to `patches.gitImage` value. |
| 1.6 | Verify ingestion Dockerfile bakes Playwright at build (not runtime) | ✅ Done | `playwright install --with-deps chromium` runs in `RUN` layer at build time — no runtime download |
| 1.7 | Disable vLLM genesis patches (avoid GitHub clone at startup) | ✅ Done | `patches.enabled: false` confirmed as default in `vllm-server/values.yaml` — no change needed |

**Verification:** After Phase 1, deploy to a node with outbound network blocked. All pods must reach `Running` state and serve requests without errors.

---

### Phase 2: Quick Quality Wins
> Code-only changes. No new services, no new models.

| # | Task | Status | Notes |
|---|---|---|---|
| 2.1 | Fix URL deduplication — allow up to 3 chunks per source URL | ✅ Done | `ai-agent/main.py:119-129` — `seen_urls` set replaced with `url_counts` dict capped at 3 |
| 2.2 | Add query rewriting step before embedding | ✅ Done | `ai-agent/main.py` — `rewrite_query()` calls the LLM (max 64 tokens, temp=0) before embedding; strips think-tags; falls back to original query on error. |
| 2.3 | Switch to RecursiveCharacterTextSplitter (sentence-aware chunking) | ✅ Done | `ingestion/main.py` — replaced word-split loop; added `langchain-text-splitters>=0.3.0` to requirements. `CHUNK_SIZE` now means characters (not words); default 512 chars ≈ 80–100 words. |
| 2.4 | Bump Qdrant fetch count from 5 to 20 (pre-reranker pool) | ✅ Done | `ai-agent/main.py` — `run_rag_search` default `top_k` changed from 5 → 20; Phase 3 reranker trims to top-5. |

---

### Phase 3: Reranker Service (New Microservice)
> Highest quality impact item. Mirrors NVIDIA's `nemotron-rerank` service.

| # | Task | Status | Notes |
|---|---|---|---|
| 3.1 | Create `services/reranker/main.py` | ✅ Done | FastAPI + CrossEncoder; `RERANKER_MODEL` env var; CUDA auto-detected; thread-pool executor for CPU/GPU bound inference |
| 3.2 | Create `services/reranker/requirements.txt` | ✅ Done | sentence-transformers==3.0.0, torch==2.3.0 (cu121), fastapi, uvicorn, pydantic |
| 3.3 | Create `services/reranker/Dockerfile` | ✅ Done | Matches embedding Dockerfile; port 8003; cu121 PyTorch index |
| 3.4 | Create `ai-stack/charts/reranker/` Helm chart | ✅ Done | Port 8003 / NodePort 30084; modelStorage block for air-gapped use; matches embedding chart structure |
| 3.5 | Wire reranker into `ai-agent/main.py` retrieval flow | ✅ Done | `RERANKER_URL` env var (empty = disabled); after Qdrant top-20 dedup → POST /rerank → top-5; falls back to top-5 by vector score on error or if disabled |
| 3.6 | Add CI workflow `.github/workflows/build-reranker.yml` | ✅ Done | Triggers on push to `ai-stack/services/reranker/**`; matches build-embeddings.yml pattern |
| 3.7 | Update `README.md` architecture section | ✅ Done | Added reranker to Services table (port 30084); updated request flow description |

**Reranker API contract:**
```
POST /rerank
{ "query": "string", "passages": ["string", ...], "top_n": 5 }
→ [{ "index": int, "score": float, "text": "string" }]
```

---

### Phase 4: Ingestion Upgrade (Document Parsing + Hybrid Search)

| # | Task | Status | Notes |
|---|---|---|---|
| 4.1 | Replace PyMuPDF with Docling in `ingestion/main.py` | ✅ Done | Lazy-init `DocumentConverter` via `_get_docling()`; writes to temp file, runs in thread pool executor; DOCX/PPTX now supported; preserves paragraph structure in output |
| 4.2 | Wire `DOCLING_ARTIFACTS_PATH` env var into ingestion Helm chart | ✅ Done | `modelStorage` block added to ingestion chart (matches embedding/reranker pattern); when enabled, `DOCLING_ARTIFACTS_PATH` set to `<mountPath>/docling` |
| 4.3 | Add SQLite FTS5 virtual table to `ingestion.db` | ✅ Done | Schema: `point_id UNINDEXED, collection UNINDEXED, url UNINDEXED, content, title, vendor, doc_id UNINDEXED`; created in `init_db()` |
| 4.4 | Populate FTS5 table alongside every Qdrant upsert | ✅ Done | `run_pipeline()` — deletes old FTS5 rows by `doc_id` before Qdrant delete; batch-inserts new rows per upsert batch |
| 4.5 | Add BM25 search endpoint to ingestion service | ✅ Done | `GET /search/lexical?q=&collection=&limit=20`; FTS5 MATCH with BM25 rank ordering; returns `{results: [{point_id, collection, url, content, title, vendor}]}` |
| 4.6 | Add RRF fusion to `ai-agent/main.py` | ✅ Done | `_rrf_merge(k=60)` merges vector hits (with `hit["id"]`) and lexical hits by `point_id`; RRF pool feeds reranker or top-5 fallback; graceful degradation if lexical search unreachable |
| 4.7 | Update ingestion `requirements.txt` | ✅ Done | Added `docling>=2.0.0`; removed `pymupdf>=1.24.0` |

---

### Phase 4b: RAG Admin UI

| # | Task | Status | Notes |
|---|---|---|---|
| 4b.1 | Create `services/rag-admin/` microservice | ✅ Done | FastAPI + embedded single-page HTML; proxies to ingestion service |
| 4b.2 | Add `rag-admin` to docker-compose.yml | ✅ Done | Port 8005; depends on ingestion healthy |
| 4b.3 | Add CI workflow `build-rag-admin.yml` | ✅ Done | Matches existing service build pattern |

**Access:** `http://<host-ip>:8005`

**Features:** drag-and-drop file upload (PDF/DOCX/PPTX/TXT/MD), URL ingestion, deep crawl, collection management, document list with status badges, delete.

**Known limitations / future work:**
- **No authentication.** The page is accessible to anyone who can reach port 8005. Access control is currently network-level only (LAN/firewall). This is intentional for the initial internal deployment.
- **No contributor roles.** All users with LAN access can add or delete content. A future `ADMIN_KEY` env-var gate or Open WebUI JWT validation could restrict writes to authorised users.
- **No audit log.** Who ingested what is not tracked. The ingestion service already records `vendor` as a source tag; a `submitted_by` field and log table could be added without changing the API contract.

---

### Phase 4c: Image-Rich PDF Handling (Page-Aware OCR)

> **Context.** Phase 4.1 adopted IBM Docling for PDF parsing. In testing (2026-06-24) Docling failed at runtime: its `StandardPdfPipeline` unconditionally loads a ~1.5 GB layout ML model (`ds4sd/docling-models`) plus optional OCR weights from `DOCLING_ARTIFACTS_PATH`, and those artifacts were never populated — so even basic text extraction errored. Docling was first replaced with lightweight `pypdf`/`python-docx`/`python-pptx` (commit `1f60b83`) to unblock ingestion. That works for born-digital text but silently drops **image-rich PDFs** (validated network/connectivity designs that live in diagrams).
>
> **Requirement (from stakeholder).** Search must find the **right page/document**; it must **not** attempt to extract design information out of the diagram. A human opens the page image to read the validated design. This rules out a VLM (which would *describe* the design) in favour of **OCR**, which lifts the diagram's text labels (device names, IPs, VLANs, ports) so the page is findable — and is verifiable against the image, keeping the future citation story clean.
>
> **Approach.** Replace `pypdf` with PyMuPDF (`fitz`) for PDFs — needed anyway for per-page image detection and on-demand rendering. OCR is a *per-page fallback* (Tesseract, CPU-only, zero model artifacts): a page is rendered + OCR'd only when its text layer is sparse and it contains images. Every chunk gains `page` + `has_image` metadata (the foundation for page-level citations). Page images are surfaced via a lazy render endpoint — no extra files stored. **Scope: PDF only**; DOCX/PPTX keep their current text-layer extraction. No VLM, no GPU, no model downloads.

| # | Task | Status | Notes |
|---|---|---|---|
| 4c.1 | `requirements.txt`: drop `pypdf`, add `pymupdf>=1.24.0` + `pytesseract>=0.3.10` | ✅ Done | PyMuPDF wheels bundle MuPDF; Tesseract is the only system dep |
| 4c.2 | `Dockerfile`: add `tesseract-ocr` apt package | ✅ Done | Language data ships with the package — no model artifacts to download |
| 4c.3 | Per-page extraction with OCR fallback | ✅ Done | `_extract_pdf_pages()` → `list[{page, text, has_image}]`; OCR via `_ocr_page()` when `len(text.strip()) < OCR_MIN_CHARS`; `has_image` from raster images, vector drawings (>`DRAWING_IMG_THRESHOLD`), or OCR |
| 4c.4 | Page-aware pipeline | ✅ Done | `run_pipeline()` accepts `segments`; each chunk tagged with `page` + `has_image` in Qdrant payload; URL/deep-crawl pass a single `page=None` segment |
| 4c.5 | Page-image render endpoint | ✅ Done | `GET /documents/{id}/pages/{n}/image` renders from the stored file via `fitz` → PNG; no PNG persisted |
| 4c.6 | OCR config env vars | ✅ Done | `OCR_ENABLED=true`, `OCR_MIN_CHARS=100`, `OCR_DPI=200` — added to `docker-compose.yml` + ingestion chart (`config.ocr*`) |
| 4c.7 | Remove vestigial Docling wiring | ✅ Done | Removed `DOCLING_ARTIFACTS_PATH` + `modelStorage` block (compose, chart deployment/values), Docling download from `scripts/download-models.sh`, and `.env.example` docling layout note |
| 4c.8 | CI build + on-node verification | ✅ Done | Verified 2026-06-24 on bengpu1 with a born-digital Dell white paper: status `completed`, 72 chunks, lexical search returns figure-adjacent content, page-image endpoint renders PNGs. Born-digital text path confirmed; OCR-on-scanned-page path not yet exercised (this PDF had a full text layer) — spot-check later with an image-only/scanned PDF. |

**Deliberately deferred (follow-on):**
- Surfacing the page image + page citation in the chat UI (ai-agent context formatting + open-webui rendering).
- Page-level citations in **lexical** (FTS5) search — the FTS5 schema is fixed at creation; adding a `page` column needs a migration. Vector search carries `page` from day one.

**Superseded rows:** Phase 4.1 (Docling adoption) and 4.7 (`docling>=2.0.0` in requirements) are obsoleted by this phase.

---

### Phase 4d: Image + Page Citations in the Chat UI

> **Context.** Phase 4c put `doc_id`, `page`, and `has_image` on every chunk and exposed `GET /documents/{id}/pages/{n}/image`. The ai-agent retrieval (`run_rag_search`) already builds a `sources` list and appends a markdown "Sources" block to the answer, but it drops the new metadata and shows only `[vendor] title — url`. This phase surfaces **page numbers** and **inline page images** (the actual validated-design diagrams) in the answer so a human can read the source page directly.
>
> **Constraint.** Open WebUI is a third-party image — we don't modify its frontend. The only surfaces we control are the assistant message's **markdown content** and the response JSON. So citations are rendered as enriched markdown in the answer (Open WebUI renders it).
>
> **Browser-reachability decision (Option A).** Page images live on the ingestion service; the user's **browser** must load them, and it can only reach the host-mapped address (e.g. `http://<host-ip>:8002`), not the internal Docker/K8s service name. The agent is told this base via `INGESTION_PUBLIC_URL`, **default empty**. Empty → text-only page citations (graceful degradation, nothing breaks). Set it (in `.env` on the host — never committed, since it contains an internal IP) → inline diagram images appear. Chosen over committing a placeholder so a fresh deploy never renders broken image links.

| # | Task | Status | Notes |
|---|---|---|---|
| 4d.1 | Carry `doc_id` / `page` / `has_image` into `sources` | ✅ Done | `run_rag_search` (`ai-agent/main.py`) — read straight from the Qdrant payload (present since Phase 4c) |
| 4d.2 | Add `INGESTION_PUBLIC_URL` config | ✅ Done | `ai-agent/main.py`; default `""` = text-only citations |
| 4d.3 | Enriched Sources renderer | ✅ Done | `format_sources()` — dedup by `(doc_id/url, page)`; append `— p.{page}`; emit a "Referenced pages" block with inline `![](…/pages/{n}/image)` only when `has_image` + `page` + `INGESTION_PUBLIC_URL` are all present |
| 4d.4 | Wire config | ✅ Done | `docker-compose.yml` (`${INGESTION_PUBLIC_URL:-}`), `.env.example` (commented, `<host-ip>` placeholder), ai-agent Helm chart (`ingestion.publicUrl`) |
| 4d.5 | On-node verification | ⬜ Pending | After CI rebuilds `open-rag-ai-agent:latest`: pull + recreate ai-agent; set `INGESTION_PUBLIC_URL` in `.env`; ask a question that hits an image page → answer shows `p.N` and the diagram renders inline |

**Deferred (separate follow-ons):**
- Inline `[1]`/`[2]` citation markers tied to individual claims, and the verified-quote-matching layer — both belong to the citations-accuracy work, not this UI-surfacing task.

**Known limitation:** lexical-*only* hits carry no `page`/`has_image` (the FTS5 table stores neither), so they fall back to a text citation without an image. Hits found via vector search — the majority — carry full metadata.

---

### Phase 5: Validation & Benchmarking
> Confirm improvements are measurable, not just theoretical.

| # | Task | Status | Notes |
|---|---|---|---|
| 5.1 | Define evaluation dataset (20–50 Q&A pairs from internal docs) | ⬜ Pending | Cover: factual lookup, keyword-heavy, multi-doc, table data |
| 5.2 | Baseline: measure retrieval accuracy before changes | ⬜ Pending | Record hit@5 and MRR@5 |
| 5.3 | Re-measure after Phase 2 (chunking + dedup + query rewriting) | ⬜ Pending | |
| 5.4 | Re-measure after Phase 3 (reranker) | ⬜ Pending | Expect 15–25% improvement |
| 5.5 | Re-measure after Phase 4 (hybrid search + Docling) | ⬜ Pending | |
| 5.6 | Load test on target GPU: measure p95 latency per query end-to-end | ⬜ Pending | Target: <3s p95 for typical query |

---

## Key Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-23 | Use SQLite FTS5 for BM25 (not OpenSearch or Qdrant sparse) | SQLite already present in ingestion service; no new infrastructure; sufficient for medium business scale |
| 2026-06-23 | Use BAAI/bge-reranker-v2-m3 (not cross-encoder/ms-marco-MiniLM) | Better multilingual support, higher benchmark scores; marginal VRAM cost difference |
| 2026-06-23 | Use IBM Docling (not Unstructured) | Fully open-source, no API tier; better table extraction; actively maintained by IBM |
| 2026-06-24 | **Reverse Docling decision** — use PyMuPDF + Tesseract OCR | Docling requires ~1.5 GB of layout/OCR model weights even for plain text extraction; weights were never populated and ingestion errored at runtime. PyMuPDF + on-demand Tesseract OCR has zero model artifacts and fits the air-gap constraint trivially. |
| 2026-06-24 | OCR (not a VLM) for image-rich pages | Requirement is page **findability**, not design extraction. OCR lifts diagram text labels for search and is verifiable against the source image; a VLM would synthesise a description (a citation-integrity risk) and cost GPU. Human reads the actual design from the surfaced page image. |
| 2026-06-24 | `page` + `has_image` on every chunk | Establishes page-level provenance — the metadata foundation for verified citations (no documents successfully ingested yet, so no migration cost). |
| 2026-06-23 | Production target is L40S (not RTX 3090) | RTX 3090 is dev/test only; L40S (48 GB VRAM) fits 70B LLM + embedding + reranker on a single card |
| 2026-06-23 | Keep Qdrant as vector store (not migrate to OpenSearch) | Already deployed, no new infrastructure; SQLite FTS5 covers the lexical gap |

---

## Open Questions

- [ ] What LLM is the production target? (Llama 3.3 70B? Qwen3 72B? Custom fine-tune?) — affects vLLM Helm values and GPU sizing
- [ ] Is web search (SearXNG) needed in production, or will RAG be strictly document-only?
- [x] Should Docling's multimodal features (chart/image understanding) be enabled? — **Resolved 2026-06-24:** Docling removed entirely (see Phase 4c). Image-rich pages handled via Tesseract OCR for findability; no vision model. A VLM-based diagram-description path remains possible later for label-free graphics, gated behind config, ideally on the L40S.
- [ ] Will Harbor be set up for the internal container registry, or use `imagePullPolicy: Never`?
- [ ] How many L40S GPUs will be in the production cluster, and will they be in one node or spread across nodes?

---

## Component Reference

### New Services Added by This Plan

| Service | Model | Port | VRAM | Purpose |
|---|---|---|---|---|
| `reranker` | BAAI/bge-reranker-v2-m3 | 8003 | ~1.5 GB | Cross-encoder reranking of top-20 candidates |

### Existing Services (unchanged)

| Service | Model/Tech | Port | Purpose |
|---|---|---|---|
| `vllm-server` | Configurable HF model | 8000 | LLM inference |
| `embedding` | nomic-ai/nomic-embed-text-v1.5 | 8001 | Dense vector embeddings |
| `ingestion` | PyMuPDF + Tesseract OCR + SQLite + Qdrant | 8002 | Document parsing, page-aware chunking, OCR fallback, indexing |
| `ai-agent` | FastAPI orchestrator | 8000 | RAG pipeline, tool calling |
| `qdrant` | Qdrant vector DB | 6333 | Vector storage and search |
| `open-webui` | Open WebUI | 80 | User interface |
| `searxng` | SearXNG (new) | 8080 | Self-hosted web search |

### Model Storage Requirements

Run `scripts/download-models.sh` once on any internet-connected machine with your model storage mounted.
Set `MODEL_DIR` to your NFS mount point or local path (default: `/mnt/nfs/models`).

| Model | Size (approx) | Used By | Notes |
|---|---|---|---|
| `nomic-ai/nomic-embed-text-v1.5` | ~550 MB | embedding service | Downloaded and verified by download-models.sh |
| `BAAI/bge-reranker-v2-m3` | ~1.1 GB | reranker service | Downloaded and verified by download-models.sh |
| LLM (your choice) | 4–70 GB | vllm-server | Download separately; set `model.name` in `vllm-server/values.yaml` |

> The ingestion service needs **no** model artifacts as of Phase 4c — PyMuPDF bundles its parser and Tesseract language data ships with the `tesseract-ocr` apt package.

After downloading, set these env vars in your Helm deployments (Phase 1.2):
- `HF_HOME=<MODEL_DIR>/huggingface` — embedding, reranker, vllm-server (ingestion needs no model storage as of Phase 4c)

---

## Reference: NVIDIA Blueprint vs Open-RAG-Stack Component Mapping

| NVIDIA Blueprint Component | Open-RAG-Stack Equivalent |
|---|---|
| `llama-3.3-nemotron-super-49b-v1.5` | Any HuggingFace LLM via vLLM |
| `llama-nemotron-embed-1b-v2` | `nomic-ai/nomic-embed-text-v1.5` |
| `llama-nemotron-rerank-1b-v2` | `BAAI/bge-reranker-v2-m3` |
| Elasticsearch (BM25 + kNN) | Qdrant (vector) + SQLite FTS5 (BM25) |
| NeMo Retriever Extraction | IBM Docling |
| LangChain orchestration | FastAPI custom orchestrator (ai-agent) |
| LangGraph multi-hop | Future work |
| OpenTelemetry observability | Future work |
