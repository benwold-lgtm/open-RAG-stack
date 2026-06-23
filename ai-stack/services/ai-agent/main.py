import os
import httpx
import json
import asyncio
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from openai import AsyncOpenAI

app = FastAPI(title="AI Agent Service")

# ── Configuration ─────────────────────────────────────────────────────────────
VLLM_BASE_URL  = os.getenv("VLLM_BASE_URL")   # e.g. http://<gpu-node-ip>:30000/v1
VLLM_MODEL     = os.getenv("VLLM_MODEL")       # HuggingFace model ID served by vllm-server
QDRANT_URL     = os.getenv("QDRANT_URL",     "http://qdrant.qdrant.svc.cluster.local:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_URL  = os.getenv("EMBEDDING_URL",  "http://embedding.embedding.svc.cluster.local:8001")

# ── Web Search Provider ───────────────────────────────────────────────────────
# Choose one provider. Uncomment the relevant block and update run_web_search()
# below to call your chosen provider's API. Only one should be active at a time.
#
# Option 1: Brave Search  (https://brave.com/search/api/ — requires API key)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
BRAVE_URL     = "https://api.search.brave.com/res/v1/web/search"
#
# Option 2: SearXNG — self-hosted, no API key required  (https://docs.searxng.org)
# Set SEARXNG_URL to your SearXNG service address and use ?q=<query>&format=json
# SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng.searxng.svc.cluster.local:8080")
#
# Option 3: Serper — Google Search results via API  (https://serper.dev)
# SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# SERPER_URL     = "https://google.serper.dev/search"
#
# Option 4: Tavily — AI-optimized search API  (https://tavily.com)
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# TAVILY_URL     = "https://api.tavily.com/search"

# ── Web Search (Brave) ────────────────────────────────────────────────────────
# Replace this function body with your chosen provider's API call.
async def run_web_search(query: str) -> str:
    if not BRAVE_API_KEY:
        return "Error: web search API key not configured."

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {
        "q": query,
        "count": 5,
        "text_decorations": False,
        "search_lang": "en"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                BRAVE_URL,
                headers=headers,
                params=params,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for r in data.get("web", {}).get("results", []):
                results.append(
                    f"Title: {r.get('title', '')}\n"
                    f"URL: {r.get('url', '')}\n"
                    f"Summary: {r.get('description', '')}\n"
                )

            return "\n---\n".join(results) if results else "No results found."

        except httpx.HTTPError as e:
            return f"Search error: {str(e)}"

# ── RAG Search ───────────────────────────────────────────────────────────────
async def run_rag_search(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    qdrant_headers = {"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        embed_resp = await client.post(
            f"{EMBEDDING_URL}/v1/embeddings/query",
            json={"input": query}
        )
        embed_resp.raise_for_status()
        query_vector = embed_resp.json()["data"][0]["embedding"]

        coll_resp = await client.get(f"{QDRANT_URL}/collections", headers=qdrant_headers)
        coll_resp.raise_for_status()
        collections = [c["name"] for c in coll_resp.json()["result"]["collections"]]

        if not collections:
            return "No ingested documents found.", []

        all_hits: list[tuple[float, dict]] = []
        for collection in collections:
            try:
                resp = await client.post(
                    f"{QDRANT_URL}/collections/{collection}/points/search",
                    headers=qdrant_headers,
                    json={"vector": query_vector, "limit": top_k, "with_payload": True},
                )
                if resp.status_code == 200:
                    for hit in resp.json().get("result", []):
                        all_hits.append((hit["score"], hit["payload"]))
            except Exception:
                continue

    if not all_hits:
        return "No relevant documents found.", []

    all_hits.sort(key=lambda x: x[0], reverse=True)

    url_counts: dict[str, int] = {}
    top_hits: list[tuple[float, dict]] = []
    for score, payload in all_hits:
        url = payload.get("url", "")
        if url_counts.get(url, 0) < 3:
            url_counts[url] = url_counts.get(url, 0) + 1
            top_hits.append((score, payload))
        if len(top_hits) >= top_k:
            break

    parts = []
    sources = []
    for _, payload in top_hits:
        parts.append(
            f"[{payload.get('title', '')} | {payload.get('vendor', '')} | {payload.get('url', '')}]\n"
            f"{payload.get('content', '')}"
        )
        sources.append({
            "url":         payload.get("url", ""),
            "title":       payload.get("title", ""),
            "vendor":      payload.get("vendor", ""),
            "chunk_index": payload.get("chunk_index", 0),
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

    # Pattern 3: bare JSON {"name": ..., "arguments": {...}}
    if not tool_calls:
        match = re.search(r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{.*?\})\}', content, re.DOTALL)
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
- Do NOT fabricate quotes or specific details not present in the retrieved context."""

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

# ── OpenAI-Compatible Endpoint ────────────────────────────────────────────────
@app.post("/v1/chat/completions")
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

        if sources:
            sources_md = "\n\n---\n**Sources:**\n" + "\n".join(
                f"- [{s['vendor']}] {s['title']} — {s['url']}"
                for s in sources
            )
            final_response += sources_md

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
            "sources": sources
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
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": VLLM_MODEL,
            "object": "model",
            "owned_by": "ai-agent"
        }]
    }
