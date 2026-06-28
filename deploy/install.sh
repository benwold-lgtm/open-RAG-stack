#!/usr/bin/env bash
# Deploys (or upgrades) all open-RAG-stack services via Helm.
# Idempotent: safe to re-run to pick up values.yaml or chart changes.
#
# Prerequisites (run scripts/bootstrap.sh first, or provide them yourself):
#   - Namespaces, secrets, and storage classes must already exist.
#   - kubectl + helm configured for the target cluster.
#
# This uses plain `helm upgrade --install` so the stack deploys to ANY
# Kubernetes cluster with no GitOps controller required. If you prefer a
# GitOps workflow, point ArgoCD / Flux / Rancher Fleet at ai-stack/charts/
# instead of running this script.
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
die()  { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHARTS_DIR="${REPO_ROOT}/ai-stack/charts"

command -v helm &>/dev/null || die "'helm' not found in PATH"

# release:namespace — ordered so data/inference layers come up before the agent
RELEASES=(
  "qdrant:qdrant"
  "embedding:embedding"
  "vllm-server:ai-stack"
  "ingestion:ingestion"
  "ai-agent:ai-agent"
  "chat-ui:chat-ui"
)

echo
echo "╔═══════════════════════════════════════════╗"
echo "║      open-RAG-stack  —  Helm Deploy       ║"
echo "╚═══════════════════════════════════════════╝"
echo

for entry in "${RELEASES[@]}"; do
  name="${entry%%:*}"
  ns="${entry##*:}"
  info "Deploying ${name} → namespace ${ns}..."
  helm upgrade --install "${name}" "${CHARTS_DIR}/${name}" \
    --namespace "${ns}" \
    --create-namespace
  ok "  ${name} deployed"
done

echo
echo "─────────────────────────────────────────────────────────────────"
ok "All services deployed. Note: vLLM takes a few minutes to load the model."
echo
echo "  Watch pods come up:"
echo "    kubectl get pods -A -w"
echo "─────────────────────────────────────────────────────────────────"
