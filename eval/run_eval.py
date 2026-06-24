#!/usr/bin/env python3
"""Phase 5.3/5.4 — retrieval metrics with ablation flags.

Reads eval/dataset.jsonl (curated gold set). For each question, runs the
retrieval pipeline and checks whether the gold chunk (point_id) is retrieved.

Metrics:
  hit@5   — gold chunk is in the final (post-rerank) top-5 context
  mrr@5   — 1/rank of the gold chunk within the top-5 (0 if absent)
  hit@20  — gold chunk is anywhere in the pre-rerank pool

Single-gold proxy: a relevant-but-not-gold chunk counts as a miss, so absolute
numbers understate quality — but it's consistent across configs, so the
ablation deltas are meaningful.

Usage:
    python3 eval/run_eval.py                 # full system
    python3 eval/run_eval.py --no-rerank     # single ablation
    python3 eval/run_eval.py --matrix        # full + every ablation, as a table
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))
import _util as u  # noqa: E402


def load_dataset(path):
    if not os.path.exists(path):
        print(f"Dataset not found: {path}\nRun gen_eval_set.py and curate first.", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(dataset, model, *, use_rewrite, use_hybrid, use_rerank):
    n = hit5 = hit20 = 0
    mrr5 = 0.0
    for item in dataset:
        gold = str(item["gold_point_id"])
        try:
            final_ids, pool_ids = u.retrieve(
                item["question"], model=model,
                use_rewrite=use_rewrite, use_hybrid=use_hybrid, use_rerank=use_rerank,
            )
        except Exception as e:
            print(f"  retrieve failed for {item['question']!r}: {e}", file=sys.stderr)
            continue
        n += 1
        if gold in pool_ids:
            hit20 += 1
        top5 = final_ids[:5]
        if gold in top5:
            hit5 += 1
            mrr5 += 1.0 / (top5.index(gold) + 1)
    if n == 0:
        return {"n": 0, "hit@5": 0.0, "mrr@5": 0.0, "hit@20": 0.0}
    return {"n": n, "hit@5": hit5 / n, "mrr@5": mrr5 / n, "hit@20": hit20 / n}


CONFIGS = [
    ("full",        dict(use_rewrite=True,  use_hybrid=True,  use_rerank=True)),
    ("-rewrite",    dict(use_rewrite=False, use_hybrid=True,  use_rerank=True)),
    ("-hybrid",     dict(use_rewrite=True,  use_hybrid=False, use_rerank=True)),
    ("-rerank",     dict(use_rewrite=True,  use_hybrid=True,  use_rerank=False)),
    ("dense-only",  dict(use_rewrite=True,  use_hybrid=False, use_rerank=False)),
]


def print_table(rows):
    print(f"\n{'config':<12} {'n':>4} {'hit@5':>8} {'mrr@5':>8} {'hit@20':>8}")
    print("-" * 44)
    for name, m in rows:
        print(f"{name:<12} {m['n']:>4} {m['hit@5']:>8.3f} {m['mrr@5']:>8.3f} {m['hit@20']:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
    ap.add_argument("--matrix", action="store_true", help="run full + all ablations")
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    model = u.vllm_model()
    print(f"Dataset: {len(dataset)} questions | Model: {model}", file=sys.stderr)

    if args.matrix:
        rows = []
        for name, flags in CONFIGS:
            print(f"Running config: {name} ...", file=sys.stderr)
            rows.append((name, evaluate(dataset, model, **flags)))
        print_table(rows)
    else:
        flags = dict(use_rewrite=not args.no_rewrite,
                     use_hybrid=not args.no_hybrid,
                     use_rerank=not args.no_rerank)
        name = "+".join(k.replace("use_", "") for k, v in flags.items() if v) or "none"
        print_table([(name, evaluate(dataset, model, **flags))])


if __name__ == "__main__":
    main()
