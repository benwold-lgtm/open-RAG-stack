#!/usr/bin/env python3
"""Phase 5.2 — generate candidate Q&A from ingested chunks (you then curate).

Scrolls Qdrant for a stratified sample of chunks (evenly spread within each
document), asks vLLM to write one question per chunk, and writes candidates to
eval/dataset.candidates.jsonl. Gold = the chunk the question was generated from.

Then curate by hand: delete weak rows, fix questions, and save the keepers to
eval/dataset.jsonl. run_eval.py reads dataset.jsonl.

Usage:
    python3 eval/gen_eval_set.py --n 60 --per-doc 3
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))
import _util as u  # noqa: E402

QGEN_SYS = (
    "You write evaluation questions for a document retrieval system. Given a "
    "passage, write ONE clear, specific question that is fully answerable using "
    "only that passage. Prefer concrete facts, configurations, or steps. Do not "
    "reference 'the passage' or 'the document'. Output only the question."
)


def looks_like_toc(text):
    return text.count(".....") > 1 or text.count("…") > 3


def collect_chunks(collections, min_chars):
    by_doc = {}
    for c in collections:
        offset = None
        while True:
            points, offset = u.qdrant_scroll(c, 256, offset)
            for p in points:
                pl = p.get("payload", {})
                text = (pl.get("content") or "").strip()
                if len(text) < min_chars or looks_like_toc(text):
                    continue
                key = pl.get("doc_id") or pl.get("url") or c
                by_doc.setdefault(key, []).append({
                    "point_id": p["id"],
                    "collection": c,
                    "doc_id": pl.get("doc_id", ""),
                    "page": pl.get("page"),
                    "title": pl.get("title", ""),
                    "chunk_index": pl.get("chunk_index", 0),
                    "text": text,
                })
            if not offset:
                break
    return by_doc


def stride_pick(items, n):
    """Pick n items evenly spread across the list (deterministic)."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def pick(by_doc, per_doc, target):
    chosen = []
    for chunks in by_doc.values():
        chunks.sort(key=lambda x: x["chunk_index"])
        chosen.extend(stride_pick(chunks, per_doc))
    return stride_pick(chosen, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="target number of candidates")
    ap.add_argument("--per-doc", type=int, default=3, help="max chunks sampled per document")
    ap.add_argument("--min-chars", type=int, default=150, help="skip chunks shorter than this")
    ap.add_argument("--collection", default=None, help="limit to one collection (default: all)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dataset.candidates.jsonl"))
    args = ap.parse_args()

    model = u.vllm_model()
    collections = [args.collection] if args.collection else u.list_collections()
    if not collections:
        print("No collections found — nothing ingested?", file=sys.stderr)
        sys.exit(1)
    print(f"Model: {model}\nCollections: {collections}", file=sys.stderr)

    by_doc = collect_chunks(collections, args.min_chars)
    selected = pick(by_doc, args.per_doc, args.n)
    print(f"Sampled {len(selected)} chunks from {len(by_doc)} documents.", file=sys.stderr)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ch in enumerate(selected, 1):
            try:
                q = u._strip_think(
                    u.vllm_chat(
                        [{"role": "system", "content": QGEN_SYS},
                         {"role": "user", "content": ch["text"]}],
                        model=model, max_tokens=64, temperature=0.2,
                    )
                ).strip().strip('"').strip()
            except Exception as e:
                print(f"  [{i}] generation failed: {e}", file=sys.stderr)
                continue
            if not q or "?" not in q:
                continue
            rec = {
                "question": q,
                "gold_point_id": str(ch["point_id"]),
                "gold_doc_id": ch["doc_id"],
                "gold_page": ch["page"],
                "collection": ch["collection"],
                "title": ch["title"],
                "source_text": ch["text"][:400],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            print(f"  [{i}/{len(selected)}] {q}", file=sys.stderr)

    print(
        f"\nWrote {written} candidates to {args.out}\n"
        f"Now curate: review each row, delete weak ones, fix questions, and save "
        f"the keepers (target 20-50) to eval/dataset.jsonl",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
