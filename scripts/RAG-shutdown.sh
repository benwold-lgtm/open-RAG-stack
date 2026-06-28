#!/usr/bin/env bash
# Shuts down the RAG stack — scales every workload to 0 to free GPU and RAM.
# Storage, secrets, and Helm releases are untouched.
# Run RAG-startup.sh to bring everything back online.
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# Explicit namespace list — intentionally scoped to avoid touching other workloads
RAG_NAMESPACES=(ai-stack ai-agent embedding ingestion chat-ui qdrant)

echo
echo "╔═══════════════════════════════════════════╗"
echo "║      open-RAG-stack  —  RAG Shutdown       ║"
echo "╚═══════════════════════════════════════════╝"
echo

# Scale all RAG deployments to 0
info "Scaling all RAG deployments to 0..."
for ns in "${RAG_NAMESPACES[@]}"; do
  count=$(kubectl get deployments -n "$ns" --no-headers 2>/dev/null | wc -l)
  if [[ "$count" -gt 0 ]]; then
    kubectl scale deployment --all --replicas=0 -n "$ns"
    ok "  Scaled to 0: $ns"
  else
    warn "  No deployments in namespace $ns (skipping)"
  fi
done

echo
echo "─────────────────────────────────────────────────────────────────"
ok "RAG lab suspended. GPU and RAM are now free."
echo
echo "  Verify all RAG pods are stopped:"
echo "    kubectl get pods -n ai-stack -n ai-agent -n embedding -n ingestion -n chat-ui -n qdrant"
echo
echo "  Bring the RAG lab back up:"
echo "    ./scripts/RAG-startup.sh"
echo "─────────────────────────────────────────────────────────────────"
