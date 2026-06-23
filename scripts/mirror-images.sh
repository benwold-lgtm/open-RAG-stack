#!/usr/bin/env bash
# mirror-images.sh — Pull all stack container images and push to an internal registry.
# Run this once from an internet-connected machine before air-gapped deployment.
#
# Requires: docker, access to push to your internal registry
#
# Usage:
#   REGISTRY=<host>/<project> ./scripts/mirror-images.sh
#
# Example (Harbor):
#   REGISTRY=harbor.example.com/open-rag ./scripts/mirror-images.sh
#
# The script prints the --set overrides to apply when running helm install/upgrade.
# Save those to a values override file or copy to ai-stack/registry-values.example.yaml.

set -euo pipefail

REGISTRY="${REGISTRY:-}"
if [ -z "$REGISTRY" ]; then
  echo "Error: REGISTRY env var is required."
  echo "Usage: REGISTRY=harbor.example.com/open-rag $0"
  exit 1
fi

mirror() {
  local SRC="$1" DST_NAME="$2"
  local DST="${REGISTRY}/${DST_NAME}"
  echo "Pulling: ${SRC}"
  docker pull "${SRC}"
  docker tag  "${SRC}" "${DST}"
  docker push "${DST}"
  echo "Pushed:  ${DST}"
  echo ""
}

echo "Mirroring all open-RAG-stack images to: ${REGISTRY}"
echo "================================================================"
echo ""

# ── Custom images (built from this repo's CI) ─────────────────────────────────
# These are re-built by the repo's GitHub Actions workflows on every push to main.
# Re-run this script (or configure your registry to proxy GHCR) after each release.
mirror "ghcr.io/benwold-lgtm/open-rag-ai-agent:latest"    "open-rag-ai-agent:latest"
mirror "ghcr.io/benwold-lgtm/open-rag-embedding:latest"   "open-rag-embedding:latest"
mirror "ghcr.io/benwold-lgtm/open-rag-ingestion:latest"   "open-rag-ingestion:latest"
mirror "ghcr.io/benwold-lgtm/open-rag-reranker:latest"    "open-rag-reranker:latest"

# ── Third-party images (pinned tags) ─────────────────────────────────────────
mirror "ghcr.io/open-webui/open-webui:v0.9.6"             "open-webui:v0.9.6"
mirror "qdrant/qdrant:v1.9.0"                             "qdrant:v1.9.0"
mirror "searxng/searxng:latest"                           "searxng:latest"
mirror "vllm/vllm-openai:v0.9.2"                         "vllm-openai:v0.9.2"

# ── Utility images ────────────────────────────────────────────────────────────
# Only needed when vllm-server patches.enabled: true (Qwen3 Genesis patches).
mirror "alpine/git:latest"                                "alpine-git:latest"

echo "================================================================"
echo "All images mirrored to ${REGISTRY}"
echo ""
echo "Apply these values when running helm install/upgrade for each chart:"
echo ""
echo "  ai-agent chart (-n ai-agent):"
echo "    --set image.repository=${REGISTRY}/open-rag-ai-agent"
echo "    --set image.pullPolicy=IfNotPresent"
echo ""
echo "  embedding chart (-n embedding):"
echo "    --set image.repository=${REGISTRY}/open-rag-embedding"
echo "    --set image.pullPolicy=IfNotPresent"
echo ""
echo "  ingestion chart (-n ingestion):"
echo "    --set image.repository=${REGISTRY}/open-rag-ingestion"
echo "    --set image.pullPolicy=IfNotPresent"
echo ""
echo "  reranker chart (-n reranker):"
echo "    --set image.repository=${REGISTRY}/open-rag-reranker"
echo "    --set image.pullPolicy=IfNotPresent"
echo ""
echo "  open-webui chart (-n open-webui):"
echo "    --set image.repository=${REGISTRY}/open-webui"
echo "    --set image.pullPolicy=IfNotPresent"
echo ""
echo "  qdrant chart (-n qdrant):"
echo "    --set image.repository=${REGISTRY}/qdrant"
echo "    --set image.pullPolicy=IfNotPresent"
echo ""
echo "  searxng chart (-n searxng):"
echo "    --set image.repository=${REGISTRY}/searxng"
echo "    --set image.pullPolicy=IfNotPresent"
echo ""
echo "  vllm-server chart (-n vllm):"
echo "    --set image.repository=${REGISTRY}/vllm-openai"
echo "    --set image.pullPolicy=IfNotPresent"
echo "    --set patches.gitImage=${REGISTRY}/alpine-git:latest"
echo ""
echo "Or copy ai-stack/registry-values.example.yaml, fill in your REGISTRY,"
echo "and apply with: helm install <release> <chart> -f registry-values.yaml"
