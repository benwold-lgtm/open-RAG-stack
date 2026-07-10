#!/usr/bin/env bash
# Validates that the environment meets the prerequisites for open-RAG-stack.
# Run this before bootstrap.sh to catch common failure modes early.
# Usage: ./scripts/prereq-check.sh [--gpu-node <hostname>]
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[PASS]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC}  $*"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }

FAILURES=0
GPU_NODE="${GPU_NODE:-}"

# Parse --gpu-node flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-node) GPU_NODE="$2"; shift 2 ;;
    *) echo "Usage: $0 [--gpu-node <hostname>]"; exit 1 ;;
  esac
done

echo
echo "╔═══════════════════════════════════════════╗"
echo "║   open-RAG-stack  —  Prerequisite Check   ║"
echo "╚═══════════════════════════════════════════╝"
echo

# ── CLI tools ──────────────────────────────────────────────────────────────────
info "Checking required CLI tools..."
for cmd in kubectl helm; do
  if command -v "$cmd" &>/dev/null; then
    ok "$cmd found ($(${cmd} version --client 2>/dev/null | head -1))"
  else
    fail "$cmd not found in PATH — install it before running bootstrap.sh"
  fi
done

# ── Cluster connectivity ────────────────────────────────────────────────────────
info "Checking cluster connectivity..."
if kubectl cluster-info &>/dev/null 2>&1; then
  SERVER=$(kubectl cluster-info 2>/dev/null | head -1 | sed 's/.*is running at //')
  ok "Cluster reachable: ${SERVER}"
else
  fail "Cannot reach cluster — check your kubeconfig (KUBECONFIG env or ~/.kube/config)"
fi

# ── Kubernetes version ─────────────────────────────────────────────────────────
info "Checking Kubernetes version..."
K8S_VERSION=$(kubectl version -o json 2>/dev/null | \
  python3 -c "import json,sys; print(json.load(sys.stdin).get('serverVersion',{}).get('gitVersion',''))" 2>/dev/null || true)
if [[ -n "$K8S_VERSION" ]]; then
  MINOR=$(echo "$K8S_VERSION" | sed 's/v[0-9]*\.\([0-9]*\).*/\1/')
  if [[ "$MINOR" -ge 24 ]]; then
    ok "Kubernetes ${K8S_VERSION} (>= 1.24 required)"
  else
    fail "Kubernetes ${K8S_VERSION} — version 1.24 or later required"
  fi
else
  warn "Could not determine server version — continuing"
fi

# ── Nodes ──────────────────────────────────────────────────────────────────────
info "Checking nodes..."
NODE_COUNT=$(kubectl get nodes --no-headers 2>/dev/null | wc -l | tr -d ' ')
READY_COUNT=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready" || true)
if [[ "$READY_COUNT" -gt 0 ]]; then
  ok "${READY_COUNT}/${NODE_COUNT} nodes Ready"
else
  fail "No nodes in Ready state"
fi

# ── NVIDIA device plugin ────────────────────────────────────────────────────────
info "Checking NVIDIA device plugin..."
if kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system &>/dev/null 2>&1; then
  ok "NVIDIA device plugin daemonset found in kube-system"
else
  fail "NVIDIA device plugin not found — vLLM will not schedule"
  warn "  Install: https://github.com/NVIDIA/k8s-device-plugin"
fi

# ── GPU resources ──────────────────────────────────────────────────────────────
info "Checking GPU resources across nodes..."
GPU_NODES=$(kubectl get nodes -o json 2>/dev/null | \
  python3 -c "
import json, sys
nodes = json.load(sys.stdin)['items']
gpu_nodes = [(n['metadata']['name'], n['status'].get('capacity', {}).get('nvidia.com/gpu', '0'))
             for n in nodes if int(n['status'].get('capacity', {}).get('nvidia.com/gpu', '0')) > 0]
for name, count in gpu_nodes:
    print(f'{name}: {count} GPU(s)')
" 2>/dev/null || true)

if [[ -n "$GPU_NODES" ]]; then
  while IFS= read -r line; do ok "GPU node — $line"; done <<< "$GPU_NODES"
else
  fail "No nodes expose nvidia.com/gpu resource — is the NVIDIA device plugin running and has it labeled nodes?"
fi

# ── GPU node specified ─────────────────────────────────────────────────────────
if [[ -n "$GPU_NODE" ]]; then
  info "Checking specified GPU node: ${GPU_NODE}..."
  if kubectl get node "$GPU_NODE" &>/dev/null 2>&1; then
    GPU_MEM=$(kubectl get node "$GPU_NODE" -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "0")
    if [[ "$GPU_MEM" -gt 0 ]]; then
      ok "Node ${GPU_NODE} has ${GPU_MEM} GPU(s) visible"
    else
      warn "Node ${GPU_NODE} exists but reports 0 GPUs — device plugin may not have started yet"
    fi
    # Disk space check
    DISK_INFO=$(kubectl get node "$GPU_NODE" -o jsonpath='{.status.capacity.ephemeral-storage}' 2>/dev/null || echo "")
    if [[ -n "$DISK_INFO" ]]; then
      ok "Node ${GPU_NODE} ephemeral storage capacity: ${DISK_INFO}"
      warn "  Ensure at least 560 Gi free — vLLM PVC requests 500 Gi; allow headroom for OS + other PVCs"
    fi
  else
    fail "Node ${GPU_NODE} not found in cluster"
  fi
else
  warn "No --gpu-node specified — skipping node-specific checks"
  warn "  Re-run with: ./scripts/prereq-check.sh --gpu-node <your-node-hostname>"
fi

# ── StorageClass: local-path ────────────────────────────────────────────────────
info "Checking local-path StorageClass..."
if kubectl get storageclass local-path &>/dev/null 2>&1; then
  ok "StorageClass 'local-path' exists"
else
  warn "StorageClass 'local-path' not found — bootstrap.sh will install it automatically"
fi

# ── DNS resolution (cluster-internal) ─────────────────────────────────────────
info "Checking cluster DNS..."
if kubectl run dns-test --image=busybox:1.36 --restart=Never --rm -it \
     --command -- nslookup kubernetes.default.svc.cluster.local &>/dev/null 2>&1; then
  ok "Cluster DNS resolves kubernetes.default.svc.cluster.local"
else
  warn "DNS check skipped or timed out — verify CoreDNS is running: kubectl get pods -n kube-system | grep coredns"
fi

# ── Internet egress (HuggingFace) ──────────────────────────────────────────────
info "Checking internet egress to HuggingFace..."
if curl -sf --max-time 5 "https://huggingface.co" -o /dev/null 2>/dev/null; then
  ok "HuggingFace (https://huggingface.co) reachable from this host"
  warn "  Note: reachability from *pods* depends on your cluster's egress policy"
else
  warn "HuggingFace unreachable from this host — vLLM model download will fail without egress"
  warn "  If running air-gapped, pre-download models and mount them manually"
fi

# ── Chart values still on placeholders ─────────────────────────────────────────
info "Checking chart values for unedited placeholders..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLACEHOLDER_FILES=$(grep -rl '<your-' "${SCRIPT_DIR}/../ai-stack/charts" --include=values.yaml 2>/dev/null || true)
if [[ -n "$PLACEHOLDER_FILES" ]]; then
  warn "These values.yaml files still contain '<your-...>' placeholders:"
  while IFS= read -r f; do warn "  ${f#"${SCRIPT_DIR}/../"}"; done <<< "$PLACEHOLDER_FILES"
  warn "  Pods will not schedule until nodeSelector points at a real node. While editing,"
  warn "  also set vllm-server model.name and ai-agent vllm.baseUrl/model (they default"
  warn "  to empty) — see README 'Quick start' and 'Choosing a model'."
else
  ok "No unedited placeholders in chart values"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo
echo "─────────────────────────────────────────────────────────────────"
if [[ "$FAILURES" -eq 0 ]]; then
  echo -e "${GREEN}All checks passed.${NC} You are ready to run ./scripts/bootstrap.sh"
else
  echo -e "${RED}${FAILURES} check(s) failed.${NC} Resolve the issues above before running bootstrap.sh"
fi
echo "─────────────────────────────────────────────────────────────────"
