#!/usr/bin/env bash
# Starts the RAG stack — scales all workloads back to 1.
# Run this after RAG-shutdown.sh. To pick up chart or values.yaml changes
# instead of just resuming, run ./deploy/install.sh.
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

NODE_IP="${NODE_IP:-<your-gpu-node-ip>}"

RAG_NAMESPACES=(ai-stack ai-agent embedding ingestion open-webui qdrant)

echo
echo "╔═══════════════════════════════════════════╗"
echo "║      open-RAG-stack  —  RAG Startup       ║"
echo "╚═══════════════════════════════════════════╝"
echo

# Scale all RAG deployments back to 1
info "Scaling all RAG deployments to 1..."
for ns in "${RAG_NAMESPACES[@]}"; do
  count=$(kubectl get deployments -n "$ns" --no-headers 2>/dev/null | wc -l)
  if [[ "$count" -gt 0 ]]; then
    kubectl scale deployment --all --replicas=1 -n "$ns"
    ok "  Scaled to 1: $ns"
  else
    warn "  No deployments in namespace $ns (skipping)"
  fi
done

echo
echo "─────────────────────────────────────────────────────────────────"
ok "RAG lab starting up. Note: vLLM takes a few minutes to load the model."
echo
echo "  Watch pods come up:"
echo "    kubectl get pods -A -w"
echo
echo "  Service endpoints (once pods reach Running):"
echo "    Open-WebUI   →  http://${NODE_IP}:30080"
echo "    vLLM API     →  http://${NODE_IP}:30000/v1"
echo "    ai-agent     →  http://${NODE_IP}:30081"
echo "    Qdrant REST  →  http://${NODE_IP}:30333"
echo "─────────────────────────────────────────────────────────────────"
