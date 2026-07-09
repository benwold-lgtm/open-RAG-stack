# Contributing

Thanks for your interest in open-RAG-stack. Contributions are welcome, with one honest
caveat up front.

## No SLA — community-maintained

This is a personal project shared in the hope it's useful. It is maintained on a
**best-effort, volunteer basis with no service-level agreement**: issues and pull
requests may take a while, or may be declined if they don't fit the project's scope.
There is no warranty (see [LICENSE](LICENSE)) and no commercial support. If you depend on
this in production, plan to maintain your own fork.

## Ways to help

- **Bugs** — open an issue with the bug template (logs, versions, repro steps).
- **Features** — open a feature-request issue first to discuss fit before building.
- **Security** — do **not** use issues; see [SECURITY.md](SECURITY.md).
- **Docs** — fixes to the README / setup steps are especially appreciated.

## Development setup

The fastest loop is single-node Docker Compose; Kubernetes (Helm) mirrors it:

```bash
cp .env.example .env          # set MODEL_DIR and LLM_MODEL (see comments)
docker compose up -d          # builds chat-ui locally, pulls the rest
```

Each service is a small FastAPI app under `ai-stack/services/`; shared auth lives in
`ai-stack/lib/rag_auth`. Helm charts are under `ai-stack/charts/`.

## Tests

Python tests run with `pytest`. Install the relevant requirements (plus
`ai-stack/lib/rag_auth[test]`) into a venv, then from each directory:

```bash
ai-stack/lib/rag_auth         # pytest      (shared auth framework)
ai-stack/services/chat-ui     # pytest
ai-stack/services/ai-agent    # pytest      (needs the service requirements installed)
ai-stack/services/ingestion   # pytest      (needs the service requirements installed)
ai-stack/services/rag-admin   # pytest
```

CI runs all five suites automatically on every pull request and push to `main`
(`.github/workflows/test.yml`) — a PR should be green there before review.

For chart changes, validate rendering with `helm template <chart> ai-stack/charts/<chart>`
(also with any feature flag you touched, e.g. `--set networkPolicy.enabled=true`).

## Pull requests

- Keep PRs small and focused; one concern per PR.
- Describe what you changed and **how you tested it**.
- Match the surrounding code style — don't reformat unrelated code.
- Update the README / docs when behavior or config changes.
- Don't commit secrets, internal IPs, or hostnames (this repo is public).
