#!/usr/bin/env python3
"""Phase 5.5 — latency / load test.

Fires the eval questions at a fixed concurrency and reports latency percentiles.
Two modes, reported separately because they answer different questions:

  retrieval — just the RAG retrieval pipeline (embed -> Qdrant -> hybrid ->
              rerank). This is what the <3s p95 target really applies to.
  e2e       — the full ai-agent /v1/chat/completions (retrieval + LLM tool loop
              + answer generation). Generation-bound on the 8B/3090, so expect
              this to dominate; it informs the L40S sizing call.

Usage:
    python3 eval/load_test.py --mode both --concurrency 4 --requests 40
"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import _util as u  # noqa: E402


def load_questions(path):
    if not os.path.exists(path):
        return ["How does the system handle indexing?"]  # fallback if no dataset yet
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line)["question"])
    return qs or ["How does the system handle indexing?"]


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_mode(name, fn, questions, concurrency, requests_n):
    tasks = [questions[i % len(questions)] for i in range(requests_n)]
    lat, errors = [], 0

    def timed(q):
        t0 = time.perf_counter()
        fn(q)
        return time.perf_counter() - t0

    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(timed, q) for q in tasks]
        for fut in as_completed(futures):
            try:
                lat.append(fut.result())
            except Exception as e:
                errors += 1
                print(f"  {name} request error: {e}", file=sys.stderr)
    wall = time.perf_counter() - wall0

    print(f"\n== {name} ==  (concurrency={concurrency}, requests={requests_n}, errors={errors})")
    if lat:
        print(f"  count {len(lat)}  mean {sum(lat)/len(lat):.2f}s  "
              f"p50 {pct(lat,50):.2f}s  p95 {pct(lat,95):.2f}s  p99 {pct(lat,99):.2f}s  "
              f"min {min(lat):.2f}s  max {max(lat):.2f}s")
        print(f"  throughput {len(lat)/wall:.2f} req/s over {wall:.1f}s wall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
    ap.add_argument("--mode", choices=["retrieval", "e2e", "both"], default="both")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=40)
    args = ap.parse_args()

    questions = load_questions(args.dataset)
    model = u.vllm_model()
    print(f"Questions: {len(questions)} | Model: {model}", file=sys.stderr)

    def retrieval_fn(q):
        u.retrieve(q, model=model)

    def e2e_fn(q):
        u.post_json(
            f"{u.AGENT_URL}/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": q}]},
            timeout=180,
        )

    if args.mode in ("retrieval", "both"):
        run_mode("retrieval", retrieval_fn, questions, args.concurrency, args.requests)
    if args.mode in ("e2e", "both"):
        run_mode("e2e", e2e_fn, questions, args.concurrency, args.requests)


if __name__ == "__main__":
    main()
