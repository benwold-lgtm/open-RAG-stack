import os
import httpx
import json
import asyncio
import re
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from openai import AsyncOpenAI
from rag_auth import RoleScopes, Principal, AUTH_LOCAL
from rag_auth.authenticator import StaticKeyAuthenticator
from rag_auth.deps import make_authenticate_request, require_scope, verify_boot_config
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="AI Agent Service")

# Prometheus metrics at GET /metrics — open like /health (only request counts/latencies, no
# secrets). Network-gate it the same way as the rest of the data plane.
Instrumentator().instrument(app).expose(app)

# ── Authentication (shared service token via rag_auth) ────────────────────────
# ai-agent sits behind the trust boundary; callers (chat-ui, scripts) present a
# shared SERVICE_TOKEN as a bearer credential. Fail-closed in production unless an
# operator explicitly opts into anonymous access on a trusted, isolated LAN.
ENVIRONMENT     = os.getenv("ENVIRONMENT",     "development")
SERVICE_TOKEN   = os.getenv("SERVICE_TOKEN",   "")
ALLOW_ANONYMOUS = os.getenv("ALLOW_ANONYMOUS", "false").lower() in ("1", "true", "yes")

CHAT_USE = "chat:use"
_ROLES = RoleScopes({"admin": {CHAT_USE}, "viewer": {CHAT_USE}})
_keys: dict[str, Principal] = {}
if SERVICE_TOKEN:
    _keys[SERVICE_TOKEN] = Principal(
        subject="key:service", scopes=_ROLES.scopes_for_role("admin"), auth_method=AUTH_LOCAL
    )
_authn = make_authenticate_request(StaticKeyAuthenticator(_keys))

for _w in verify_boot_config(
    production=ENVIRONMENT == "production",
    any_auth_enabled=bool(SERVICE_TOKEN),
    allow_anonymous=ALLOW_ANONYMOUS,
):
    print(f"[ai-agent] WARN: {_w}", flush=True)

# Enforce auth only when a token is configured AND anonymous access isn't allowed.
_AUTH_ACTIVE = bool(SERVICE_TOKEN) and not ALLOW_ANONYMOUS

def _guard(scope: str):
    return [Depends(_authn), Depends(require_scope(scope))] if _AUTH_ACTIVE else []

# Bearer header for outbound calls to other guarded services (e.g. ingestion FTS).
_SERVICE_AUTH = {"Authorization": f"Bearer {SERVICE_TOKEN}"} if SERVICE_TOKEN else {}

# ── Configuration ─────────────────────────────────────────────────────────────
VLLM_BASE_URL  = os.getenv("VLLM_BASE_URL")   # e.g. http://<gpu-node-ip>:30000/v1
VLLM_MODEL     = os.getenv("VLLM_MODEL")       # HuggingFace model ID served by vllm-server
QDRANT_URL     = os.getenv("QDRANT_URL",     "http://qdrant.qdrant.svc.cluster.local:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_URL  = os.getenv("EMBEDDING_URL",  "http://embedding.embedding.svc.cluster.local:8001")
RERANKER_URL   = os.getenv("RERANKER_URL",  "")  # empty = reranking disabled
INGESTION_URL  = os.getenv("INGESTION_URL", "http://ingestion.ingestion.svc.cluster.local:8002")
# Browser-reachable base for page-image links (e.g. http://<host-ip>:8002).
# Empty = text-only page citations (no inline images). See ENHANCEMENT-PLAN Phase 4d.
INGESTION_PUBLIC_URL = os.getenv("INGESTION_PUBLIC_URL", "")
# Verified citations: ask the model for verbatim supporting quotes and check each
# against the retrieved chunk text. Empty/false = off. See ENHANCEMENT-PLAN Phase 4e.
CITE_VERIFY = os.getenv("CITE_VERIFY", "true").lower() in ("1", "true", "yes")
# Query rewriting before retrieval. Phase 5 ablation showed it is net-negative for
# dense retrieval with nomic (keyword text embeds worse than the natural question),
# so default off. See ENHANCEMENT-PLAN Phase 5/6.
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "false").lower() in ("1", "true", "yes")

# ── Web Search Provider ───────────────────────────────────────────────────────
# Set WEB_SEARCH_PROVIDER in the ai-agent Helm chart values.yaml. Options:
#   searxng  — self-hosted, no API key (deploy ai-stack/charts/searxng)
#   brave    — Brave Search API — requires BRAVE_API_KEY secret
#   serper   — Google results via Serper — requires SERPER_API_KEY secret
#   tavily   — AI-optimized search — requires TAVILY_API_KEY secret
#   none     — web search disabled (default)
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "none")
BRAVE_API_KEY  = os.getenv("BRAVE_API_KEY", "")
SEARXNG_URL    = os.getenv("SEARXNG_URL", "http://searxng.searxng.svc.cluster.local:8080")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

async def run_web_search(query: str) -> str:
    if WEB_SEARCH_PROVIDER == "brave":
        return await _search_brave(query)
    if WEB_SEARCH_PROVIDER == "searxng":
        return await _search_searxng(query)
    if WEB_SEARCH_PROVIDER == "serper":
        return await _search_serper(query)
    if WEB_SEARCH_PROVIDER == "tavily":
        return await _search_tavily(query)
    return "Web search is disabled. Set WEB_SEARCH_PROVIDER in the ai-agent Helm values to enable it."

async def _search_brave(query: str) -> str:
    if not BRAVE_API_KEY:
        return "Error: BRAVE_API_KEY not set."
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
                params={"q": query, "count": 5, "text_decorations": False, "search_lang": "en"},
                timeout=10.0,
            )
            r.raise_for_status()
            results = r.json().get("web", {}).get("results", [])
            return "\n---\n".join(
                f"Title: {x.get('title','')}\nURL: {x.get('url','')}\nSummary: {x.get('description','')}"
                for x in results
            ) or "No results found."
        except httpx.HTTPError as e:
            return f"Search error: {e}"

async def _search_searxng(query: str) -> str:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "categories": "general"},
                timeout=10.0,
            )
            r.raise_for_status()
            results = r.json().get("results", [])[:5]
            return "\n---\n".join(
                f"Title: {x.get('title','')}\nURL: {x.get('url','')}\nSummary: {x.get('content','')}"
                for x in results
            ) or "No results found."
        except httpx.HTTPError as e:
            return f"Search error: {e}"

async def _search_serper(query: str) -> str:
    if not SERPER_API_KEY:
        return "Error: SERPER_API_KEY not set."
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
                timeout=10.0,
            )
            r.raise_for_status()
            results = r.json().get("organic", [])
            return "\n---\n".join(
                f"Title: {x.get('title','')}\nURL: {x.get('link','')}\nSummary: {x.get('snippet','')}"
                for x in results
            ) or "No results found."
        except httpx.HTTPError as e:
            return f"Search error: {e}"

async def _search_tavily(query: str) -> str:
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY not set."
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
                timeout=10.0,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            return "\n---\n".join(
                f"Title: {x.get('title','')}\nURL: {x.get('url','')}\nSummary: {x.get('content','')}"
                for x in results
            ) or "No results found."
        except httpx.HTTPError as e:
            return f"Search error: {e}"

# ── Query Rewriting ───────────────────────────────────────────────────────────
async def rewrite_query(query: str) -> str:
    """Rewrite the user's query into a keyword-rich retrieval query via the LLM.
    Falls back to the original query on any error."""
    try:
        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
        response = await client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the following question as a concise, keyword-rich search query "
                        "optimized for document retrieval. Output only the rewritten query, no explanation."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        rewritten = strip_think_tags(response.choices[0].message.content or "").strip()
        return rewritten if rewritten else query
    except Exception:
        return query

# ── RRF merge ────────────────────────────────────────────────────────────────
def _rrf_merge(
    vector_hits: list[tuple[float, int, dict]],
    lexical_hits: list[dict],
    k: int = 60,
) -> list[tuple[float, dict]]:
    """Merge vector and lexical ranked lists via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, (_, pid, payload) in enumerate(vector_hits, start=1):
        key = str(pid)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        payloads[key] = payload

    for rank, item in enumerate(lexical_hits, start=1):
        key = str(item.get("point_id", ""))
        if not key:
            continue
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in payloads:
            payloads[key] = {
                "url":         item.get("url", ""),
                "title":       item.get("title", ""),
                "vendor":      item.get("vendor", ""),
                "collection":  item.get("collection", ""),
                "content":     item.get("content", ""),
                "chunk_index": 0,
            }

    return sorted(
        [(scores[k], payloads[k]) for k in scores],
        key=lambda x: x[0],
        reverse=True,
    )

# ── Diagram intent ────────────────────────────────────────────────────────────
_DIAGRAM_INTENT_RE = re.compile(
    r'\b(diagram|architecture|topology|schematic|figure|illustration|drawing|'
    r'blueprint|reference\s+design|visual|picture|chart|layout)\b',
    re.IGNORECASE,
)


def _diagram_intent(query: str) -> bool:
    """True when the user is asking for a visual (a diagram/architecture/figure). Used to keep
    the document's own figure pages from being buried by the prose-favoring reranker."""
    return bool(_DIAGRAM_INTENT_RE.search(query or ""))


# ── RAG Search ───────────────────────────────────────────────────────────────
async def run_rag_search(query: str, top_k: int = 20) -> tuple[str, list[dict]]:
    search_query = await rewrite_query(query) if QUERY_REWRITE else query
    qdrant_headers = {"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        embed_resp = await client.post(
            f"{EMBEDDING_URL}/v1/embeddings/query",
            json={"input": search_query}
        )
        embed_resp.raise_for_status()
        query_vector = embed_resp.json()["data"][0]["embedding"]

        coll_resp = await client.get(f"{QDRANT_URL}/collections", headers=qdrant_headers)
        coll_resp.raise_for_status()
        collections = [c["name"] for c in coll_resp.json()["result"]["collections"]]

        if not collections:
            return "No ingested documents found.", []

        vector_hits: list[tuple[float, int, dict]] = []
        for collection in collections:
            try:
                resp = await client.post(
                    f"{QDRANT_URL}/collections/{collection}/points/search",
                    headers=qdrant_headers,
                    json={"vector": query_vector, "limit": top_k, "with_payload": True},
                )
                if resp.status_code == 200:
                    for hit in resp.json().get("result", []):
                        vector_hits.append((hit["score"], hit["id"], hit["payload"]))
            except Exception:
                continue

        lexical_hits: list[dict] = []
        try:
            lex_resp = await client.get(
                f"{INGESTION_URL}/search/lexical",
                params={"q": search_query, "limit": top_k},
                headers=_SERVICE_AUTH,
            )
            if lex_resp.status_code == 200:
                lexical_hits = lex_resp.json().get("results", [])
        except Exception:
            pass

    if not vector_hits and not lexical_hits:
        return "No relevant documents found.", []

    vector_hits.sort(key=lambda x: x[0], reverse=True)

    all_hits = _rrf_merge(vector_hits, lexical_hits) if lexical_hits \
               else [(score, payload) for score, _, payload in vector_hits]

    url_counts: dict[str, int] = {}
    top_hits: list[tuple[float, dict]] = []
    for score, payload in all_hits:
        url = payload.get("url", "")
        if url_counts.get(url, 0) < 3:
            url_counts[url] = url_counts.get(url, 0) + 1
            top_hits.append((score, payload))
        if len(top_hits) >= top_k:
            break

    # Rerank when available. Request the FULL ordered list (not just the top 5) so the
    # diagram-intent reserve step below has demoted-but-relevant candidates to promote from.
    if RERANKER_URL and top_hits:
        passages = [payload.get("content", "") for _, payload in top_hits]
        try:
            async with httpx.AsyncClient(timeout=30.0) as rc:
                r = await rc.post(
                    f"{RERANKER_URL}/rerank",
                    json={"query": query, "passages": passages, "top_n": len(passages)},
                )
                r.raise_for_status()
                ranked = r.json()
            ordered = [(item["score"], top_hits[item["index"]][1]) for item in ranked]
        except Exception:
            ordered = top_hits
    else:
        ordered = top_hits

    top_hits = ordered[:5]
    # Diagram intent: when the user explicitly asks for a diagram/architecture/figure, make sure
    # the document's own figure surfaces. The cross-encoder reranker favors prose over short
    # caption units, so if no image-bearing unit made the top 5, reserve the last slot for the
    # best-ranked caption/diagram page — format_sources renders that page to the user as an image.
    if _diagram_intent(query):
        _is_visual = lambda p: p.get("kind") == "caption" or p.get("has_image")
        if not any(_is_visual(p) for _, p in top_hits):
            best_visual = next(((s, p) for s, p in ordered if _is_visual(p)), None)
            if best_visual:
                top_hits = top_hits[:4] + [best_visual]

    parts = []
    sources = []
    for _, payload in top_hits:
        page = payload.get("page")
        loc = f" | p.{page}" if page is not None else ""
        parts.append(
            f"[{payload.get('title', '')} | {payload.get('vendor', '')}{loc} | {payload.get('url', '')}]\n"
            f"{payload.get('content', '')}"
        )
        sources.append({
            "url":         payload.get("url", ""),
            "title":       payload.get("title", ""),
            "vendor":      payload.get("vendor", ""),
            "chunk_index": payload.get("chunk_index", 0),
            "doc_id":      payload.get("doc_id", ""),
            "page":        payload.get("page"),
            "has_image":   payload.get("has_image", False),
            "content":     payload.get("content", ""),
        })

    return "\n\n---\n\n".join(parts), sources


# ── Think-tag stripper ────────────────────────────────────────────────────────
def strip_think_tags(text: str) -> str:
    if '</think>' in text:
        after = text.split('</think>', 1)[-1].strip()
        return after if after else text.strip()
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# ── Tool call parser ──────────────────────────────────────────────────────────
def extract_tool_calls(content: str) -> list:
    tool_calls = []

    # Pattern 1: [TOOL_CALLS] array format
    match = re.search(r'\[TOOL_CALLS\]\s*(\[.*?\])', content, re.DOTALL)
    if match:
        try:
            calls = json.loads(match.group(1))
            if isinstance(calls, list):
                tool_calls.extend(calls)
        except json.JSONDecodeError:
            pass

    # Pattern 2: <tool_call> XML tag format
    if not tool_calls:
        for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL):
            try:
                tool_calls.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass

    # Pattern 3: bare JSON {"name": ..., "arguments": {...}} — tolerate whitespace and
    # newlines after the braces so pretty-printed / ```json-fenced tool calls also match
    # (e.g. Qwen2.5-VL emits multi-line fenced JSON, not compact <tool_call> tags).
    if not tool_calls:
        match = re.search(r'\{\s*"name":\s*"(\w+)",\s*"arguments":\s*(\{.*?\})\s*\}', content, re.DOTALL)
        if match:
            try:
                tool_calls.append({
                    "name": match.group(1),
                    "arguments": json.loads(match.group(2))
                })
            except json.JSONDecodeError:
                pass

    return tool_calls

# ── Core agent loop ───────────────────────────────────────────────────────────
async def run_agent(
    messages: list,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> tuple[str, list[dict]]:
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

    system_prompt = """You are a helpful assistant with access to two tools:

1. rag_search — Search the ingested documentation and knowledge base.
   Use this first for any questions about products, vendors, or technical topics.
   Format: <tool_call>{"name": "rag_search", "arguments": {"query": "your search query"}}</tool_call>

2. web_search — Search the live internet for current information.
   Use this for recent news, current events, weather, pricing, or when rag_search returns no useful results.
   Format: <tool_call>{"name": "web_search", "arguments": {"query": "your search query"}}</tool_call>

Always try rag_search first. Use web_search when the knowledge base lacks the answer or for live/current information.

When formulating your answer after receiving tool results:
- Answer ONLY using the information returned by the tools.
- If the retrieved context does not contain enough information to answer, say so explicitly.
- Do NOT use your training knowledge to supplement the retrieved context.
- Do NOT fabricate quotes or specific details not present in the retrieved context.
- When the context includes a figure or diagram caption (text beginning "Figure N." or "Table N."),
  the user is automatically shown that page as an image below your answer. Tell them which
  document and page contains the diagram; never claim you are unable to display images."""

    if CITE_VERIFY:
        system_prompt += """

After your answer, on a new line, output a block of the verbatim quotes from the
retrieved context that support your key claims, in exactly this format:
[[CITATIONS]]
"first verbatim quote copied word-for-word from the context"
"second verbatim quote"
Include 1 to 4 quotes, each at most ~30 words, copied exactly from the retrieved
context. If a claim has no verbatim support in the context, do not invent a quote."""

    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    max_iterations = 3
    iteration = 0
    all_sources: list[dict] = []

    while iteration < max_iterations:
        iteration += 1

        response = await client.chat.completions.create(
            model=model or VLLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=top_p,           # uncomment to override; default is model-server default (usually 1.0)
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # Qwen3: disables chain-of-thought
        )
        # For models that emit <think>...</think> reasoning blocks (e.g. Qwen3, DeepSeek-R1),
        # uncomment the line below to strip them from the response before tool parsing.
        # content = strip_think_tags(response.choices[0].message.content or "")
        content = response.choices[0].message.content or ""

        tool_calls = extract_tool_calls(content)

        if not tool_calls:
            return content, all_sources

        tool_results = []
        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("arguments", {})
            query = tool_args.get("query", "")

            if tool_name == "rag_search":
                result_text, sources = await run_rag_search(query)
                all_sources.extend(sources)
                tool_results.append(f"Knowledge base results for '{query}':\n{result_text}")
            elif tool_name == "web_search":
                result_text = await run_web_search(query)
                tool_results.append(f"Web search results for '{query}':\n{result_text}")
            else:
                tool_results.append(f"Unknown tool: {tool_name}")

        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": "Here are the results:\n\n" + "\n\n".join(tool_results) + "\n\nPlease provide a helpful answer based on these results."
        })

    return "I was unable to complete the request after multiple attempts.", all_sources

# ── Request/Response Models ───────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

# ── Source / citation formatting ──────────────────────────────────────────────
def format_sources(sources: list[dict]) -> str:
    """Render the Sources block with page citations as clickable links, plus inline page
    images when an image-bearing PDF page is available and INGESTION_PUBLIC_URL is configured.
    The title links to the web URL for scraped docs, or to the served original file
    (/documents/{id}/file) for uploaded/watch-folder docs when INGESTION_PUBLIC_URL is set.
    Deduplicates by (doc_id/url, page) so repeated chunks collapse to one entry."""
    seen = set()
    lines = []
    images = []
    for s in sources:
        doc_id = s.get("doc_id", "")
        page = s.get("page")
        key = (doc_id or s.get("url", ""), page)
        if key in seen:
            continue
        seen.add(key)

        url = s.get("url", "")
        title = s.get("title") or url or "source"
        if url.startswith("http"):
            link = url                                                  # scraped web page
        elif INGESTION_PUBLIC_URL and doc_id:
            link = f"{INGESTION_PUBLIC_URL}/documents/{doc_id}/file"     # served original file
        else:
            link = ""
        label = f"[{title}]({link})" if link else title
        cite = f"- [{s.get('vendor', '')}] {label}"
        if page is not None:
            cite += f" — p.{page}"
        lines.append(cite)

        if INGESTION_PUBLIC_URL and s.get("has_image") and page is not None and doc_id:
            img_url = f"{INGESTION_PUBLIC_URL}/documents/{doc_id}/pages/{page}/image"
            images.append(f"![{s.get('title', '')} p.{page}]({img_url})")

    md = "\n\n---\n**Sources:**\n" + "\n".join(lines)
    if images:
        md += "\n\n**Referenced pages:**\n" + "\n\n".join(images)
    return md

# ── Verified citations ────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    """Lowercase, unify unicode dashes/quotes, collapse whitespace — for matching."""
    for a, b in (("‑", "-"), ("–", "-"), ("—", "-"),
                 ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"')):
        text = text.replace(a, b)
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_citations(content: str) -> tuple[str, list[str]]:
    """Split model output into (answer without the quote block, [verbatim quotes])."""
    m = re.search(r'\[\[CITATIONS\]\]\s*(.*)$', content, re.DOTALL)
    if not m:
        return content, []
    answer = content[:m.start()].rstrip()
    quotes = [q.strip() for q in re.findall(r'"([^"]+)"', m.group(1)) if q.strip()]
    return answer, quotes

def verify_citations(quotes: list[str], sources: list[dict]) -> list[dict]:
    """Deterministically check each quote against retrieved chunk text.
    Returns [{quote, verified, page}] — page is the matched source's page."""
    norm_sources = [(s, _normalize(s.get("content", ""))) for s in sources]
    results = []
    for q in quotes:
        nq = _normalize(q)
        match = next((s for s, nc in norm_sources if nq and nq in nc), None)
        results.append({
            "quote": q,
            "verified": match is not None,
            "page": match.get("page") if match else None,
        })
    return results

def format_citations(verifications: list[dict]) -> str:
    if not verifications:
        return ""
    lines = []
    for v in verifications:
        if v["verified"]:
            loc = f" — p.{v['page']}" if v.get("page") is not None else ""
            lines.append(f'- ✓ "{v["quote"]}"{loc}')
        else:
            lines.append(f'- ⚠ "{v["quote"]}" — not found in retrieved sources')
    return "\n\n**Verified quotes:**\n" + "\n".join(lines)

# ── OpenAI-Compatible Endpoint ────────────────────────────────────────────────
@app.post("/v1/chat/completions", dependencies=_guard(CHAT_USE))
async def chat_completions(request: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        final_response, sources = await run_agent(
            messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )

        citations: list[dict] = []
        if CITE_VERIFY:
            final_response, quotes = extract_citations(final_response)
            # Only verify against retrieved chunks; skip for web-search-only answers.
            if sources:
                citations = verify_citations(quotes, sources)
                if citations:
                    final_response += format_citations(citations)

        if sources:
            final_response += format_sources(sources)

        if request.stream:
            return StreamingResponse(
                stream_text(final_response),
                media_type="text/event-stream"
            )

        return {
            "id": "chatcmpl-agent",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "sources": sources,
            "citations": citations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_text(text: str):
    words = text.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == len(words) - 1 else word + " "
        data = {
            "id": "chatcmpl-agent",
            "object": "chat.completion.chunk",
            "model": VLLM_MODEL,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.01)
    yield "data: [DONE]\n\n"

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "vllm_url": VLLM_BASE_URL}

# ── Models Endpoint ───────────────────────────────────────────────────────────
@app.get("/v1/models", dependencies=_guard(CHAT_USE))
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": VLLM_MODEL,
            "object": "model",
            "owned_by": "ai-agent"
        }]
    }
