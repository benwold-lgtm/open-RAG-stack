# Open-RAG-Stack Enhancement Plan
## Goal: NVIDIA Blueprint Parity — Fully Internal Deployment

**Last Updated:** 2026-06-23
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
| PDF parsing | ⚠️ PyMuPDF (flat text) | ✅ IBM Docling (structure-aware) |
| Chunking strategy | ⚠️ word-based | ✅ RecursiveCharacterTextSplitter |
| Multi-chunk per source | ❌ 1 chunk/URL max | ✅ 3 chunks/URL max |
| Query rewriting | ❌ none | ✅ uses existing LLM |
| Web search | ⚠️ Brave/Serper/Tavily (cloud) | ✅ SearXNG (self-hosted) |
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
| 1.3 | Replace Brave/Serper/Tavily with SearXNG | ⬜ Pending | SearXNG option already in `ai-agent/main.py` (commented out at line 31) |
| 1.4 | Deploy SearXNG to K8s (new Helm chart) | ⬜ Pending | Can restrict to internal sources if needed, or disable web_search entirely |
| 1.5 | Mirror container images to internal registry | ⬜ Pending | Harbor is the recommended path; fallback is `imagePullPolicy: Never` on pre-pulled nodes |
| 1.6 | Verify ingestion Dockerfile bakes Playwright at build (not runtime) | ⬜ Pending | Lines 12-13 of `ingestion/Dockerfile` — confirm no runtime download |
| 1.7 | Disable vLLM genesis patches (avoid GitHub clone at startup) | ⬜ Pending | Set `patches.enabled: false` in `vllm-server/values.yaml` — default is already false |

**Verification:** After Phase 1, deploy to a node with outbound network blocked. All pods must reach `Running` state and serve requests without errors.

---

### Phase 2: Quick Quality Wins
> Code-only changes. No new services, no new models.

| # | Task | Status | Notes |
|---|---|---|---|
| 2.1 | Fix URL deduplication — allow up to 3 chunks per source URL | ✅ Done | `ai-agent/main.py:119-129` — `seen_urls` set replaced with `url_counts` dict capped at 3 |
| 2.2 | Add query rewriting step before embedding | ⬜ Pending | `ai-agent/main.py` — ~15 lines, uses existing LLM |
| 2.3 | Switch to RecursiveCharacterTextSplitter (sentence-aware chunking) | ⬜ Pending | `ingestion/main.py:110-119` — add `langchain-text-splitters` to requirements |
| 2.4 | Bump Qdrant fetch count from 5 to 20 (pre-reranker pool) | ⬜ Pending | `ai-agent/main.py` — single constant change; reranker in Phase 3 picks top-5 |

---

### Phase 3: Reranker Service (New Microservice)
> Highest quality impact item. Mirrors NVIDIA's `nemotron-rerank` service.

| # | Task | Status | Notes |
|---|---|---|---|
| 3.1 | Create `services/reranker/main.py` | ⬜ Pending | FastAPI, loads `BAAI/bge-reranker-v2-m3` via CrossEncoder |
| 3.2 | Create `services/reranker/requirements.txt` | ⬜ Pending | sentence-transformers, fastapi, torch, uvicorn |
| 3.3 | Create `services/reranker/Dockerfile` | ⬜ Pending | Match pattern of embedding service Dockerfile |
| 3.4 | Create `ai-stack/charts/reranker/` Helm chart | ⬜ Pending | Same structure as embedding chart; set `nodeSelector` to your GPU node |
| 3.5 | Wire reranker into `ai-agent/main.py` retrieval flow | ⬜ Pending | After Qdrant top-20 → call reranker → take top-5 → assemble LLM context |
| 3.6 | Add CI workflow `.github/workflows/build-reranker.yml` | ⬜ Pending | Match existing build workflows |
| 3.7 | Update `README.md` architecture section | ⬜ Pending | Add reranker to component list and diagram |

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
| 4.1 | Replace PyMuPDF with Docling in `ingestion/main.py` | ⬜ Pending | Lines 158-176; add DOCX/PPTX support; update SUPPORTED_EXTENSIONS |
| 4.2 | Wire `DOCLING_ARTIFACTS_PATH` env var into ingestion Helm chart | ⬜ Pending | Point to `<MODEL_DIR>/docling` on your model storage |
| 4.3 | Add SQLite FTS5 virtual table to `ingestion.db` | ⬜ Pending | Schema: `point_id, collection, content, title, vendor` |
| 4.4 | Populate FTS5 table alongside every Qdrant upsert | ⬜ Pending | `ingestion/main.py` — add INSERT to FTS5 table in upsert loop |
| 4.5 | Add BM25 search endpoint to ingestion service | ⬜ Pending | `GET /search/lexical?q=...&collection=...&limit=20` |
| 4.6 | Add RRF fusion to `ai-agent/main.py` | ⬜ Pending | Call lexical search + vector search in parallel; merge with RRF (k=60) |
| 4.7 | Update ingestion `requirements.txt` | ⬜ Pending | Add `docling>=2.0.0`; remove `pymupdf>=1.24.0` |

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
| 2026-06-23 | Production target is L40S (not RTX 3090) | RTX 3090 is dev/test only; L40S (48 GB VRAM) fits 70B LLM + embedding + reranker on a single card |
| 2026-06-23 | Keep Qdrant as vector store (not migrate to OpenSearch) | Already deployed, no new infrastructure; SQLite FTS5 covers the lexical gap |

---

## Open Questions

- [ ] What LLM is the production target? (Llama 3.3 70B? Qwen3 72B? Custom fine-tune?) — affects vLLM Helm values and GPU sizing
- [ ] Is web search (SearXNG) needed in production, or will RAG be strictly document-only?
- [ ] Should Docling's multimodal features (chart/image understanding) be enabled? Requires a vision model alongside the LLM.
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
| `ingestion` | Docling + SQLite + Qdrant | 8002 | Document parsing, chunking, indexing |
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
| Docling layout models | ~1.5 GB | ingestion service | Downloaded and verified by download-models.sh |
| LLM (your choice) | 4–70 GB | vllm-server | Download separately; set `model.name` in `vllm-server/values.yaml` |

After downloading, set these env vars in your Helm deployments (Phase 1.2):
- `HF_HOME=<MODEL_DIR>/huggingface` — all services
- `DOCLING_ARTIFACTS_PATH=<MODEL_DIR>/docling` — ingestion service only

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
