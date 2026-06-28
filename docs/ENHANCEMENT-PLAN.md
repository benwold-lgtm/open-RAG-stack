# Open-RAG-Stack Enhancement Plan
## Goal: NVIDIA Blueprint Parity — Fully Internal Deployment

**Last Updated:** 2026-06-26
**Status:** Phases 1–6 complete & validated on the GPU node (RTX 3090). **Phase 7 (rag-admin UX & hardening) in progress.**
**Maintained by:** open-RAG-stack contributors

> **Program closeout (2026-06-24).** Shipped & validated end-to-end: hybrid search + reranker (Phases 2–4), page-aware PDF parsing with OCR fallback (4c), image + page citations in chat (4d), verified citations (4e), and an eval/benchmark harness (Phase 5). Phase 6 tuned retrieval: chunk size 256 and query-rewrite-off lifted page-level hit@5 from 0.770 → 0.820. Retrieval is well-tuned; the next quality lever is the LLM (generation), which is a hardware/model call, not a code one. See **Deployment Notes** for the Compose↔K8s parity audit.
>
> **Phase 7 opened (2026-06-26).** Operating the rag-admin UI surfaced gaps — no K8s chart, no move/rename for collections, opaque ingestion failures, no table sort/filter, no auth — now an active phase (see below). **Still parked:** contextual retrieval (4.6/6.4/6.6 — diminishing returns vs effort) and the L40S production model.

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
| 4c.8 | CI build + on-node verification | ✅ Done | Verified 2026-06-24 on the GPU node with a born-digital Dell white paper: status `completed`, 72 chunks, lexical search returns figure-adjacent content, page-image endpoint renders PNGs. Born-digital text path confirmed; OCR-on-scanned-page path not yet exercised (this PDF had a full text layer) — spot-check later with an image-only/scanned PDF. |

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
| 4d.5 | On-node verification | ✅ Done | Verified 2026-06-24 on the GPU node: agent answer shows `— p.N` citations; only the `has_image:true` page (p.11) emits a "Referenced pages" inline image; `sources[]` carries `doc_id`/`page`/`has_image`; dedup collapses chunks to distinct pages. Final visual confirmation = image renders inline in Open WebUI. (Also fixed: vLLM `gpu-memory-utilization` 0.85→0.70 default — it shares GPU 0 with embedding+reranker; see `VLLM_GPU_UTIL`.) |

**Deferred (separate follow-ons):**
- Inline `[1]`/`[2]` citation markers tied to individual claims, and the verified-quote-matching layer — both belong to the citations-accuracy work, not this UI-surfacing task.

**Known limitation:** lexical-*only* hits carry no `page`/`has_image` (the FTS5 table stores neither), so they fall back to a text citation without an image. Hits found via vector search — the majority — carry full metadata.

---

### Phase 4e: Verified Citations (quote → source match)

> **Context.** Phase 4d surfaces *which* page an answer came from. This phase adds a defensible accuracy check: the model emits the verbatim quotes it relied on, and the agent **deterministically verifies** each against the retrieved chunk text. A quote that doesn't match the sources is **flagged** (not silently trusted) — catching fabricated or paraphrased-as-quote citations. This is the "don't trust, verify" property that the proprietary `citations` API feature gives for free but vLLM does not.
>
> **Decisions (stated, not silently chosen):**
> - **Failure = flag, never delete.** Unverified quotes get a ⚠ marker; the answer text is never mangled. Transparency over silent edits for accuracy-critical use.
> - **Match = deterministic normalized substring** (lowercased, collapsed whitespace, unified unicode dashes/quotes) against the retrieved chunk content. No fuzzy matching (avoids false "verified").
> - **Graceful degradation.** If the model emits no quote block (the 8B is not perfectly reliable at this), output is unchanged — no verification section, nothing breaks.
> - **Toggle** `CITE_VERIFY` (default on) to disable if noisy on a small model.

| # | Task | Status | Notes |
|---|---|---|---|
| 4e.1 | Carry chunk `content` into `sources` | ✅ Done | `run_rag_search` — match corpus (from the Qdrant payload) |
| 4e.2 | `CITE_VERIFY` config + prompt instruction | ✅ Done | When on, appends a `[[CITATIONS]]` block instruction to the system prompt asking for verbatim supporting quotes |
| 4e.3 | Parse + verify helpers | ✅ Done | `extract_citations()`, `verify_citations()` (normalized substring match per source → page attribution), `format_citations()` (✓/⚠ render) |
| 4e.4 | Integrate in `chat_completions` | ✅ Done | Strips quote block from the visible answer; appends "Verified quotes" (✓ "…" — p.N / ⚠ "…" — not found); adds `citations` to response JSON. Verification skipped for web-only answers (no RAG sources) |
| 4e.5 | Wire config | ✅ Done | `docker-compose.yml` + ai-agent Helm chart (`citeVerify`/`CITE_VERIFY`) |
| 4e.6 | On-node verification | ✅ Done | Verified 2026-06-24 on the GPU node: 3 verbatim quotes all ✓, each attributed to the correct matched page (incl. p.11 vs p.10 disambiguation); structured `citations[]` in JSON. The 8B quoted verbatim here — ⚠ path (paraphrase/fabrication) is the trivial else-branch, will surface naturally when the model rewords. |

**Known limitation:** small models often *paraphrase* rather than quote verbatim, which will show as ⚠ even when the claim is sound — the flag means "not verbatim-verifiable," not "false." Larger models (L40S) quote more faithfully. The `Sources`/page citations from Phase 4d remain the always-on provenance; verified quotes are an additional, best-effort integrity layer.

---

### Phase 5: Validation & Benchmarking
> Confirm improvements are measurable, not just theoretical.

> **Reframed 2026-06-24.** The original 5.2–5.5 measured retrieval "before/after each phase," but Phases 2–4 are all already shipped — there is no pre-phase baseline left to capture. Instead we measure the **current full system** and **ablate** components using the toggles already built (`RERANKER_URL`, hybrid lexical, query rewriting) to attribute each one's contribution. Decisions: eval set built **hybrid** (LLM-generated candidates → human-curated); metrics = **retrieval + latency** (answer-faithfulness deferred). Gold = the chunk a question was generated from (single-gold hit@k — a relevant-but-not-gold chunk counts as a miss, slightly understating quality, fine for relative/ablation comparison). Primary k=5 (final context post-rerank); also report @20 (pre-rerank pool).
>
> **Tooling:** stdlib-only Python scripts in `eval/` (committed). Generated Q&A and results stay **local and gitignored** — they contain internal-doc content, which must not land in the public repo.

| # | Task | Status | Notes |
|---|---|---|---|
| 5.1 | Corpus inventory & sanity check | ✅ Done | 2026-06-24: 28 docs, 3,248 chunks, all `completed` (none `unchanged`/zero). "Fast ingest" confirmed legit — GPU embedding + born-digital text extraction. |
| 5.2 | Generate + curate eval set (hybrid) | ✅ Done | 61-question curated set (`eval/dataset.jsonl`, gitignored), LLM-generated from sampled chunks across the 28 docs |
| 5.3 | Retrieval metrics — current full system | ✅ Done | Full system: **hit@5 0.738, MRR@5 0.634, hit@20 0.738** (see Results below) |
| 5.4 | Ablations | ✅ Done | Reranker is the only lever with lift (+0.066 hit@5 / +0.098 MRR). Hybrid & query-rewrite neutral on this synthetic set (caveat: paraphrase questions favour dense retrieval). See Results |
| 5.5 | Latency | ✅ Done | retrieval p95 **0.77s** (target <3s crushed); e2e p95 **8.75s** / p50 5.49s — generation-bound (8B, ~2–3 LLM calls/query, concurrency 4) |
| 5.6 | Record results + sizing call | ✅ Done | See Conclusions below — 3090 fine for retrieval + single-user; L40S warranted for answer latency/throughput/model quality |

**Results — retrieval ablation (2026-06-24, n=61, k=5):**

| config | hit@5 | MRR@5 | hit@20 |
|---|---|---|---|
| full | 0.738 | 0.634 | 0.738 |
| −rewrite | 0.738 | 0.623 | 0.738 |
| −hybrid | 0.738 | 0.634 | 0.738 |
| −rerank | 0.672 | 0.536 | 0.738 |
| dense-only | 0.672 | 0.517 | 0.738 |

Reading: (1) **reranker earns its place** — the only component whose removal moves the metric. (2) **`hit@20` is flat at 0.738 across all configs and equals full `hit@5`** → the reranker already promotes every gold-in-pool chunk into the top-5; the ceiling is **first-stage recall**, not reranking.

**Follow-up diagnostics (2026-06-24):**
- **Page-level match** (`--match page`): hit@5 0.738 → **0.770** (+0.03 only). The recall gap is *real*, not a single-gold scoring artifact — most misses are genuine, not "right page, wrong chunk."
- **Doc-level match** (`--match doc`): hit@5 **0.934**, hit@20 **0.951** — the embedder retrieves the right *document* ~93–95% of the time. The ~19-pt gap down to chunk-level (0.74) is **intra-document localization** — a chunking problem, **not** an embedder problem. Decisive diagnostic → Phase 6.
- **Larger pool** (`--top-k 40`): hit@40 = **0.738**, identical to hit@20 → widening the first-stage pool does nothing; missed chunks aren't in ranks 21–40 either.
- **Query rewriting is net-negative**: at top_k=40, `-rewrite` scored **0.754 > 0.738** full. Rewriting a natural question into keyword text embeds *worse* with nomic (trained on natural-language queries). Candidate to disable / make optional — validate on real multi-turn queries first (rewrite may still help pronoun/acronym resolution this single-shot synthetic set doesn't capture).
- **Hybrid lexical**: neutral on this synthetic set; kept (cheap, helps real keyword queries the synthetic set under-represents).

**Results — latency (2026-06-24, concurrency 4, 40 requests):**

| path | p50 | p95 | p99 | throughput |
|---|---|---|---|---|
| retrieval-only | 0.60s | 0.77s | 0.81s | 6.8 req/s |
| end-to-end | 5.49s | 8.75s | 10.48s | 0.72 req/s |

**Phase 5 conclusions & sizing call:**
- **Retrieval is fast and the GPU is not its bottleneck** — sub-second p95 on the 3090. The quality ceiling is **first-stage recall (~0.74 chunk / 0.77 page)**, which is an *embedding/chunking* problem, not a hardware or reranker one. Biggest quality lever next: better embedder and/or chunking; reranker and pool-size are already maxed.
- **Reranker: keep** (validated lift). **Query rewriting: candidate to drop** (neutral-to-negative + an extra LLM call). **Hybrid: keep** (cheap insurance).
- **Latency is generation-bound.** retrieval p95 0.77s ✓; e2e p95 8.75s ✗ vs the <3s target — entirely the 8B generating on a shared 3090.
- **3090 vs L40S:** the 3090 is fine for retrieval and single-user functional validation. **L40S is warranted for production** to fix answer latency/throughput and to run a larger LLM (which also lifts the verbatim-quote fidelity noted in Phase 4e) — driven by *generation*, not retrieval.

---

### Phase 6: Retrieval Recall — Chunking

> **Context.** Phase 5's doc-level diagnostic is decisive: the embedder retrieves the right *document* 93–95% of the time, but the right *chunk* only ~74%. The gap is **intra-document localization** — wrong chunk within the right doc — so the lever is **chunking**, not the embedder (which would be churn + an `EMBEDDING_DIM` change for no gain).
>
> **Eval methodology.** Re-ingesting re-chunks the corpus, which invalidates the chunk-level gold `point_id`s — but `gold_page`/`gold_doc_id` are stable (properties of the source). So **chunking experiments are scored at page level** (`run_eval.py --match page`). Baseline to beat: **page-level hit@5 = 0.770**; the ceiling to chase is doc-level **0.934**. Each experiment requires a full re-ingest of the corpus.

| # | Task | Status | Notes |
|---|---|---|---|
| 6.1 | Query-rewrite toggle (default off) + chunk params `.env`-tunable | ✅ Done | `QUERY_REWRITE` (ai-agent, default false) wired in compose + chart; `CHUNK_SIZE`/`CHUNK_OVERLAP` now `${..}`-overridable in compose. Phase 5 showed rewrite net-negative |
| 6.2 | Experiment A — larger chunks (1024/128) | ✅ Done (negative) | page-level hit@5 **0.770 → 0.639** — *worse*. Bigger chunks embed as blurrier averages → right chunk ranks lower; `hit@20` also fell (0.770→0.656). Gradient says go smaller, not bigger. |
| 6.3 | Experiment A2 — smaller chunks (256/64) | ✅ Done (win) | rewrite-off page-level hit@5 **0.770 → 0.820** (mrr 0.655→0.714, hit@20 0.836). Monotonic: 256 > 512 > 1024. **Locked 256/64 as the default** (compose + chart). Caveat: smaller chunks mildly inflate the page metric (more chunks/page); true arbiter is answer quality (deferred). Rewrite hurts *more* at 256 (full 0.705 vs -rewrite 0.820) — reconfirms toggle-off |
| 6.4 | Experiment B — contextual chunk headers (code) | ⏸️ Parked | Title-prefix is constant within a doc → won't help intra-doc localization; real contextual retrieval = LLM-per-chunk (~7.4k calls) or heading extraction. High effort, hard to measure cleanly (page-metric inflation), diminishing returns vs the 256/rewrite-off win. Revisit only if answer quality demands it |
| 6.5 | Customizable drop folder in compose | ✅ Done | `WATCH_DIR`/`WATCH_HOST_DIR`/`WATCH_POLL_INTERVAL` wired into compose ingestion (mount → `/mnt/watch`); documented in `.env.example`. (UI-configurable drop folder = low-priority future item) |
| 6.6 | Dense-vs-lexical rewrite routing | ⏸️ Parked | Keep query rewriting but feed the keyword rewrite only to the **lexical/BM25** leg (where keywords help) while the **dense** leg gets the natural question. Lets rewriting return as a net positive. Revisit with the future UX/changes batch |

---

### Phase 7: RAG Admin UX & Hardening

> **Context.** Phase 4b shipped the rag-admin UI (compose-only). Operating it surfaced gaps that this phase closes, combining operator-requested features with the review's minor findings.
>
> **Key design fact.** A "collection" is a real Qdrant collection (one per name — see `ingestion/main.py` `ensure_collection` / delete-by-`collection_name`), **not** a payload field. So **move** and **rename** are *vector migrations* (scroll points out → upsert into target → delete from source, plus update the SQLite `documents` row and the FTS5 `collection` column), not metadata edits. All collections share the 768-dim nomic embedding, so a move **carries existing vectors — no re-embedding**.
>
> **Auth decision (2026-06-26).** HTTP Basic Auth, single admin credential from env/secret (`ADMIN_USER` / `ADMIN_PASSWORD`), **default-off** for backward-compat. Chosen over localhost-bind (loses LAN + K8s access) and shared-token (no identity, leaks into URLs). Watch-folder ingestion is unaffected — it's gated by filesystem rights natively.

| # | Task | Status | Notes |
|---|---|---|---|
| 7.0 | Docs anonymization scrub | ✅ Done | Prior pass placeholdered IPs correctly but missed one leaked internal GPU-node hostname — now replaced with the neutral "the GPU node" across this plan + `eval/README.md`. By decision, the LICENSE copyright holder and the public GHCR image handle are kept (the handle is already exposed by the repo URL; the image paths must match where images live). Use "the GPU node (RTX 3090)" in future on-node notes. |
| 7.P1 | Helm chart for `rag-admin` | ✅ Done | `ai-stack/charts/rag-admin/` — Deployment + Service (NodePort **30085**), `INGESTION_URL`, non-root uid 1000, health probes. P6 auth-secret wiring (`auth.enabled` → `ADMIN_USER`/`ADMIN_PASSWORD` from `secretKeyRef`) included now so the chart isn't touched twice. Mirrors the reranker/ai-agent chart structure. |
| 7.P2 | Error visibility — click `failed` status → reason | ✅ Done | Frontend only; `documents.error` was already stored and returned in the list payload. Failed-status badge is now clickable → modal with the stored failure reason + source/collection/vendor. Folds in **finding #3** (`esc()` now escapes `'`; delete/error use `data-*` + event delegation instead of inline-onclick string building). |
| 7.P3 | Sort + filter the document table | ✅ Done | Client-side on the fetched array: click-to-sort headers (Source/Collection/Vendor/Type/Status/Updated, asc/desc arrows) + a free-text filter box. Folds in **finding #4** (upload-queue badge now reflects the real terminal status via `syncQueueBadges` instead of a blind 30 s auto-remove). |
| 7.P4 | Move document to another collection | ✅ Done | Ingestion `POST /documents/{id}/move` + `_copy_points()` helper: scroll points by `doc_id` (with_vectors) → ensure target → upsert (re-tag payload `collection`, preserve point IDs) → delete from source → update SQLite + FTS5 `collection`. No re-embed. rag-admin proxies it; UI adds a per-row ⇱ move button → target-collection modal. |
| 7.P5 | Rename collection | ✅ Done | Ingestion `POST /collections/{name}/rename`: create target → `_copy_points()` over the whole collection → `delete_collection` old → re-tag SQLite + FTS5. Guards: 404 if missing, 409 if target exists. O(points) (sync Qdrant calls — fine for an admin action). rag-admin proxies (URL-encoded name, 300 s timeout); UI adds a Rename control next to *+ New*. |
| 7.P6 | Basic Auth (single admin cred) | ✅ Done | `rag-admin` ASGI middleware: when `ADMIN_USER`+`ADMIN_PASSWORD` are both set, every route except `/health` requires HTTP Basic Auth (constant-time compare); unset = no auth (default, backward-compatible). Wired in compose (`.env`) and the P1 chart secret. The UI nav badge + access note reflect the live auth state. `access_roles`/`classification` (**finding #2**) still deferred. |
| 7.7 | Deep-crawl UX: in-UI help + user docs | ✅ Done | Plain-language guidance for *Add Content → Deep crawl options* in the admin UI — Max depth / Max pages / URL pattern filter, each with an inline explanation and a concrete wildcard example (`*/docs/*`); fixed the misleading regex-style placeholder. Added a **"Deep crawl explained"** section to the README (behaviour, a worked filter-example table, first-time-user tips), linked from the UI. Goal: showcase the crawl capability and lower the barrier for inexperienced self-hosters. Rolled into the P4–P6 batch. |
| 7.8 | Bulk select + bulk actions (delete / move) | ✅ Done | Document table gains a checkbox column + header **select-all** (operates on the filtered rows) and a bulk-action bar (**Move to…**, **Delete**, Clear). Selection persists across sort/filter/refresh (a `Set`, pruned when docs disappear); auto-refresh pauses during a bulk op. Frontend-only — iterates the validated single-doc `DELETE` / `…/move` endpoints sequentially and reports a summary (`Deleted N of M (k failed)`). No backend change. Branch `feat/rag-admin-bulk-actions`. |

**Branch plan:** P1+P2+P3 shipped together (`feat/rag-admin-ux-p1-p3`) — chart + frontend wins, independent of the backend migration work. Batch 2 (`feat/rag-admin-p4-p6`): the deep-crawl UX docs (7.7) plus the P4→P5 ingestion endpoints and P6 auth. P6 reuses the chart secret wiring already shipped in P1.

---

## Deployment Notes — Compose ↔ Kubernetes parity (audit 2026-06-24)

Static audit of the Helm charts vs the compose stack (not live-tested — no spare cluster). Service **images** are CI-built from the same `Dockerfile`/`main.py`, so application code + deps are identical across both; only env/config wiring can drift.

**In parity (verified):**
- ai-agent chart wires every env the service reads, including the new `INGESTION_PUBLIC_URL`, `CITE_VERIFY`, `QUERY_REWRITE`.
- ingestion chart wires `OCR_ENABLED/MIN_CHARS/DPI` and `chunkSize: 256` / `chunkOverlap: 64`; all Docling wiring removed (`DOCLING_ARTIFACTS_PATH`, `modelStorage`).
- Drop folder already supported in k8s via the ingestion `watchDir` block (compose gained the equivalent in Phase 6.5).

**GPU topology — the one place compose and k8s legitimately differ:**
- In **compose**, vLLM + embedding + reranker all share GPU 0, so `gpu-memory-utilization` must be **0.70** (the 0.85/0.90 default OOMs — see Phase 5 fix).
- In **k8s**, only `vllm-server` requests `nvidia.com/gpu: 1` (exclusive); **embedding and reranker request no GPU → they run CPU-only**. So vLLM owns the card and `gpuMemoryUtilization: 0.90` is correct there. Do **not** lower it to match compose.

**⚠️ Soundness flags for when the K8s path is exercised (needs hardware to validate):**
1. **embedding/reranker are CPU-only in k8s today.** That's schedulable and correct on a single-GPU node, but slower than compose (where they're GPU-accelerated). The documented L40S budget assumes all three on the GPU — to realize that you must enable **NVIDIA device-plugin time-slicing/MPS**, add `nvidia.com/gpu` requests to the embedding/reranker charts, **and** lower vLLM's util to ~0.70 (the compose situation). Not wired today.
2. **Set `ai-agent` `ingestion.publicUrl` to `http://<node-ip>:30083`** (the ingestion NodePort) for inline page images to load in the browser; default empty = text-only citations.
3. vLLM `model.name` and `ai-agent` `vllm.baseUrl`/`vllm.model` must be filled per deployment (no defaults).

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
| 2026-06-24 | Phase 5: keep reranker, keep hybrid, flag query-rewrite for removal | Ablation on 61-Q eval (28-doc corpus): reranker is the only lever with measurable lift; hybrid neutral but cheap; **query rewriting is net-negative** for dense retrieval with nomic (keyword text embeds worse than the natural question). Validate rewrite on real multi-turn queries before removing. |
| 2026-06-24 | Production target L40S is driven by generation, not retrieval | Retrieval p95 0.77s on the 3090 (target <3s); e2e p95 8.75s is generation-bound. Quality ceiling is first-stage recall (~0.74), an embedding/chunking problem independent of GPU. L40S buys faster/larger LLM + throughput + headroom. |
| 2026-06-24 | Chunk size 256 (was 512) | Phase 6 ablation: recall gap was intra-document localization (doc-recall 0.93 vs chunk 0.74). Smaller chunks embed more precisely → page-level hit@5 0.770 (512) → 0.820 (256); 1024 was worse (0.656). Locked 256/64 in compose + chart. |
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
