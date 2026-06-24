# Phase 5 — Retrieval Evaluation & Benchmarking

Stdlib-only Python scripts that measure the running stack. They talk to the
services **directly** (not through the ai-agent) so retrieval can be ablated
cleanly. Run them on a host that can reach the services (e.g. bengpu1).

> **Privacy:** generated Q&A and results contain your internal-doc content and
> are **gitignored** (`dataset*.jsonl`, `results*`). Only these scripts are
> committed. Do not commit the data into this public repo.

## Service URLs

Default to docker-compose host ports on localhost. Override via env vars if
needed: `EMBEDDING_URL`, `QDRANT_URL`, `INGESTION_URL`, `RERANKER_URL`,
`VLLM_URL`, `AGENT_URL`, `QDRANT_API_KEY`.

## Workflow

**5.1 — Inventory** (sanity-check what's actually ingested):
```bash
curl -s http://localhost:8002/documents \
  | python3 -c "import sys,json; d=json.load(sys.stdin)['documents']; \
print('docs:',len(d)); \
[print(x['status'], x.get('chunk_count'), x['url']) for x in d]"
```
Confirm chunk_counts are > 0 and statuses are `completed` (not all `unchanged`).

**5.2 — Generate + curate the eval set:**
```bash
python3 eval/gen_eval_set.py --n 60 --per-doc 3      # -> eval/dataset.candidates.jsonl
# Review candidates: delete weak rows, fix questions, keep 20-50.
# Save the keepers as eval/dataset.jsonl (one JSON object per line).
```
Each row: `question`, `gold_point_id`, `gold_doc_id`, `gold_page`, `collection`,
`title`, `source_text`. Only `question` and `gold_point_id` are used for scoring;
the rest are there to help you curate.

**5.3 / 5.4 — Retrieval metrics + ablations:**
```bash
python3 eval/run_eval.py --matrix     # full + every ablation as a table
python3 eval/run_eval.py --no-rerank  # one config at a time, if preferred
```
Reports `hit@5`, `mrr@5` (final post-rerank context) and `hit@N` (pre-rerank pool).

Interpreting recall on a topically-redundant corpus (separate true recall gap
from single-gold label strictness):
```bash
python3 eval/run_eval.py --matrix --match page    # credit a hit at page granularity
python3 eval/run_eval.py --matrix --top-k 40      # does a bigger pool lift recall?
```
`--match` is `point` (exact chunk, default) | `page` (same doc+page) | `doc` (same document).

**5.5 — Latency:**
```bash
python3 eval/load_test.py --mode both --concurrency 4 --requests 40
```
`retrieval` p95 is the number to compare against the <3s target; `e2e` is
generation-bound (informs the L40S decision).
