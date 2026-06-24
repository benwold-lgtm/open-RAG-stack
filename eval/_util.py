"""Shared, stdlib-only client helpers for the Phase 5 eval harness.

Talks to the running stack's services directly (not through the ai-agent) so the
retrieval pipeline can be ablated cleanly. Service URLs default to localhost host
ports (docker-compose); override via env vars if running elsewhere.
"""
import os
import re
import json
import urllib.parse
import urllib.request

EMBEDDING_URL  = os.getenv("EMBEDDING_URL",  "http://localhost:8001")
QDRANT_URL     = os.getenv("QDRANT_URL",     "http://localhost:6333")
INGESTION_URL  = os.getenv("INGESTION_URL",  "http://localhost:8002")
RERANKER_URL   = os.getenv("RERANKER_URL",   "http://localhost:8003")
VLLM_URL       = os.getenv("VLLM_URL",       "http://localhost:8000")
AGENT_URL      = os.getenv("AGENT_URL",      "http://localhost:8004")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")


# ── HTTP ──────────────────────────────────────────────────────────────────────
def _request(method, url, payload=None, headers=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get_json(url, headers=None, timeout=120):
    return _request("GET", url, None, headers, timeout)


def post_json(url, payload, headers=None, timeout=120):
    return _request("POST", url, payload, headers, timeout)


def _qdrant_headers():
    return {"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else {}


# ── Service clients ─────────────────────────────────────────────────────────────
def list_collections():
    r = get_json(f"{QDRANT_URL}/collections", headers=_qdrant_headers())
    return [c["name"] for c in r["result"]["collections"]]


def embed_query(text):
    """Query-side embedding — uses the query endpoint so the nomic 'search_query'
    prefix matches what the ai-agent does in production."""
    r = post_json(f"{EMBEDDING_URL}/v1/embeddings/query", {"input": text})
    return r["data"][0]["embedding"]


def qdrant_search(collection, vector, limit):
    coll = urllib.parse.quote(collection, safe="")
    r = post_json(
        f"{QDRANT_URL}/collections/{coll}/points/search",
        {"vector": vector, "limit": limit, "with_payload": True},
        headers=_qdrant_headers(),
    )
    return r.get("result", [])


def qdrant_scroll(collection, limit, offset=None):
    coll = urllib.parse.quote(collection, safe="")
    payload = {"limit": limit, "with_payload": True, "with_vector": False}
    if offset is not None:
        payload["offset"] = offset
    r = post_json(
        f"{QDRANT_URL}/collections/{coll}/points/scroll",
        payload, headers=_qdrant_headers(),
    )
    res = r.get("result", {})
    return res.get("points", []), res.get("next_page_offset")


def lexical_search(q, limit, collection=None):
    url = f"{INGESTION_URL}/search/lexical?q={urllib.parse.quote(q)}&limit={limit}"
    if collection:
        url += f"&collection={urllib.parse.quote(collection)}"
    try:
        return get_json(url).get("results", [])
    except Exception:
        return []


def rerank(query, passages, top_n):
    return post_json(
        f"{RERANKER_URL}/rerank",
        {"query": query, "passages": passages, "top_n": top_n},
    )


def vllm_model():
    return get_json(f"{VLLM_URL}/v1/models")["data"][0]["id"]


def vllm_chat(messages, model=None, max_tokens=256, temperature=0.0):
    r = post_json(
        f"{VLLM_URL}/v1/chat/completions",
        {"model": model or vllm_model(), "messages": messages,
         "max_tokens": max_tokens, "temperature": temperature},
    )
    return r["choices"][0]["message"]["content"]


# ── Retrieval pipeline (mirrors ai-agent run_rag_search, with ablation flags) ───
def _strip_think(text):
    if "</think>" in text:
        after = text.split("</think>", 1)[-1].strip()
        return after if after else text.strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def rewrite_query(query, model=None):
    msgs = [
        {"role": "system", "content": (
            "Rewrite the following question as a concise, keyword-rich search query "
            "optimized for document retrieval. Output only the rewritten query, no explanation."
        )},
        {"role": "user", "content": query},
    ]
    try:
        out = _strip_think(vllm_chat(msgs, model=model, max_tokens=64, temperature=0.0)).strip()
        return out or query
    except Exception:
        return query


def _rrf_merge(vector_hits, lexical_hits, k=60):
    """vector_hits: [(score, point_id, payload)]; returns [(score, pid_str, payload)]."""
    scores, payloads = {}, {}
    for rank, (_, pid, payload) in enumerate(vector_hits, start=1):
        key = str(pid)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        payloads[key] = payload
    for rank, item in enumerate(lexical_hits, start=1):
        key = str(item.get("point_id", ""))
        if not key:
            continue
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        payloads.setdefault(key, {"url": item.get("url", ""), "content": item.get("content", "")})
    return sorted(((scores[k2], k2, payloads[k2]) for k2 in scores),
                  key=lambda x: x[0], reverse=True)


def retrieve(question, *, top_k=20, final_n=5,
             use_rewrite=True, use_hybrid=True, use_rerank=True, model=None):
    """Return (final_ids, pool_ids) as lists of point_id strings.
    final_ids = post-rerank top-n; pool_ids = pre-rerank deduped pool (top_k)."""
    search_query = rewrite_query(question, model) if use_rewrite else question
    vector = embed_query(search_query)

    vhits = []
    for c in list_collections():
        try:
            for h in qdrant_search(c, vector, top_k):
                vhits.append((h["score"], h["id"], h.get("payload", {})))
        except Exception:
            continue
    vhits.sort(key=lambda x: x[0], reverse=True)

    lex = lexical_search(search_query, top_k) if use_hybrid else []
    if use_hybrid and lex:
        merged = _rrf_merge(vhits, lex)
    else:
        merged = [(s, str(pid), p) for s, pid, p in vhits]

    # cap 3 chunks per source url (mirrors ai-agent)
    url_counts, top = {}, []
    for s, pid, p in merged:
        url = p.get("url", "")
        if url_counts.get(url, 0) < 3:
            url_counts[url] = url_counts.get(url, 0) + 1
            top.append((s, pid, p))
        if len(top) >= top_k:
            break

    pool_ids = [pid for _, pid, _ in top]

    if use_rerank and RERANKER_URL and top:
        passages = [p.get("content", "") for _, _, p in top]
        try:
            ranked = rerank(question, passages, final_n)
            final = [top[item["index"]] for item in ranked]
        except Exception:
            final = top[:final_n]
    else:
        final = top[:final_n]

    return [pid for _, pid, _ in final], pool_ids
