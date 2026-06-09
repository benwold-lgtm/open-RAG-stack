#!/usr/bin/env bash
# Bootstrap script for open-RAG-stack.
# Run this on any host with kubectl + helm configured for the cluster.
# Assumes: Kubernetes cluster is up and nodes are joined. kubeconfig is set.
# Does NOT provision VMs or install Kubernetes itself.
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()  { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Config ─────────────────────────────────────────────────────────────────────
# Set these to match your environment before running.
NODE_IP="${NODE_IP:-<your-gpu-node-ip>}"   # IP of the node running AI workloads
NFS_SERVER="${NFS_SERVER:-<your-nfs-server-ip>}"
# Both nfs-client PVCs (ingestion, open-webui) and the vLLM static PV share this
# export. nfs-subdir-external-provisioner creates subdirectories within it, so
# there is no conflict with the vLLM models stored at the root.
NFS_PATH="${NFS_PATH:-/NFS_K8S_PV}"

LOCAL_PATH_MANIFEST="https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml"

# ── Pre-flight ─────────────────────────────────────────────────────────────────
echo
echo "╔═══════════════════════════════════════════╗"
echo "║       open-RAG-stack  —  Bootstrap        ║"
echo "╚═══════════════════════════════════════════╝"
echo
info "Running pre-flight checks..."
for cmd in kubectl helm; do
  command -v "$cmd" &>/dev/null || die "'$cmd' not found in PATH"
done
kubectl cluster-info &>/dev/null || die "Cannot reach cluster — check kubeconfig"
ok "Cluster reachable"

# NFS server reachability (port 2049)
info "Checking NFS server reachability (${NFS_SERVER})..."
if nc -z -w5 "${NFS_SERVER}" 2049 2>/dev/null; then
  ok "NFS server reachable"
else
  warn "NFS server ${NFS_SERVER}:2049 unreachable — vLLM and ingestion PVCs will fail to bind"
fi

# NVIDIA device plugin (required for vLLM GPU scheduling)
if kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system &>/dev/null; then
  ok "NVIDIA device plugin found"
else
  warn "NVIDIA device plugin not found in kube-system — vLLM will fail to schedule"
  warn "  Install: https://github.com/NVIDIA/k8s-device-plugin"
fi

# ── Storage: local-path-provisioner (used by qdrant) ───────────────────────────
if kubectl get storageclass local-path &>/dev/null; then
  warn "StorageClass 'local-path' already exists — skipping"
else
  info "Installing local-path-provisioner..."
  kubectl apply -f "$LOCAL_PATH_MANIFEST"
  ok "local-path-provisioner installed"
fi

# ── Storage: nfs-subdir-external-provisioner (used by ingestion, open-webui) ───
if kubectl get storageclass nfs-client &>/dev/null; then
  warn "StorageClass 'nfs-client' already exists — skipping"
else
  info "Adding nfs-subdir-external-provisioner Helm repo..."
  helm repo add nfs-subdir-external-provisioner \
    https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/ --force-update
  info "Installing nfs-client StorageClass (NFS ${NFS_SERVER}:${NFS_PATH})..."
  helm install nfs-subdir-external-provisioner \
    nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
    --namespace kube-system \
    --set nfs.server="${NFS_SERVER}" \
    --set nfs.path="${NFS_PATH}" \
    --set storageClass.name=nfs-client \
    --set storageClass.reclaimPolicy=Retain
  ok "nfs-client StorageClass installed"
fi

# ── Namespaces ─────────────────────────────────────────────────────────────────
# Namespaces and secrets must exist before `helm install` so pods can mount
# their secrets on first start.
info "Creating namespaces..."
for ns in ai-agent qdrant ai-stack ingestion embedding open-webui; do
  kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f -
done
ok "Namespaces ready"

# ── Secrets ────────────────────────────────────────────────────────────────────
prompt_secret() {
  local prompt="$1" varname="$2" value=""
  while [[ -z "$value" ]]; do
    read -rsp "  ${prompt}: " value; echo
    [[ -z "$value" ]] && warn "Value cannot be empty — try again."
  done
  printf -v "$varname" '%s' "$value"
}

prompt_optional_secret() {
  local prompt="$1" varname="$2" value=""
  read -rsp "  ${prompt}: " value; echo
  printf -v "$varname" '%s' "$value"
}

echo
echo "  ┌─────────────────────────────────────────────────────────────┐"
echo "  │  SECRET STORAGE REMINDER                                    │"
echo "  │  Store every value entered below in Bitwarden (or your      │"
echo "  │  password manager) under an entry named:                    │"
echo "  │    'open-RAG-stack bootstrap secrets'                        │"
echo "  │                                                             │"
echo "  │  The optional ghcr-pull-secret PAT (read:packages) is only  │"
echo "  │  needed if your GHCR images are private. CI itself uses the  │"
echo "  │  built-in GITHUB_TOKEN — no PAT required. Store any PAT in   │"
echo "  │  Bitwarden too.                                             │"
echo "  └─────────────────────────────────────────────────────────────┘"
echo
info "Creating Kubernetes secrets — input is hidden, press Enter after each value."
echo

if kubectl get secret ai-agent-secrets -n ai-agent &>/dev/null; then
  warn "Secret 'ai-agent-secrets' already exists — skipping"
else
  prompt_optional_secret "WEB_SEARCH_API_KEY (Brave / Serper / Tavily — press Enter to skip for SearXNG)" WEB_SEARCH_API_KEY
  prompt_secret "QDRANT_API_KEY (for ai-agent RAG access)" QDRANT_API_KEY_AGENT
  kubectl create secret generic ai-agent-secrets -n ai-agent \
    --from-literal=BRAVE_API_KEY="${WEB_SEARCH_API_KEY}" \
    --from-literal=QDRANT_API_KEY="${QDRANT_API_KEY_AGENT}"
  ok "Created secret 'ai-agent-secrets' in namespace ai-agent"
fi

if kubectl get secret qdrant-secrets -n qdrant &>/dev/null; then
  warn "Secret 'qdrant-secrets' already exists — skipping"
else
  prompt_secret "QDRANT_API_KEY (for qdrant)" QDRANT_API_KEY
  kubectl create secret generic qdrant-secrets -n qdrant \
    --from-literal=QDRANT_API_KEY="${QDRANT_API_KEY}"
  ok "Created secret 'qdrant-secrets' in namespace qdrant"
fi

if kubectl get secret hf-token-secret -n ai-stack &>/dev/null; then
  warn "Secret 'hf-token-secret' already exists — skipping"
else
  prompt_secret "HF_TOKEN (Hugging Face, for vLLM model download)" HF_TOKEN
  kubectl create secret generic hf-token-secret -n ai-stack \
    --from-literal=token="${HF_TOKEN}"
  ok "Created secret 'hf-token-secret' in namespace ai-stack"
fi

# ── GHCR pull secret (required if your packages are private) ──────────────────
echo
info "Checking whether GHCR images are accessible without auth..."
echo "  The three custom images (ai-agent, embedding, ingestion) are hosted on ghcr.io."
echo "  If you made those packages public (Settings → Package → Change visibility → Public)"
echo "  you can skip this step. Otherwise enter your GitHub username and a PAT with"
echo "  'read:packages' scope to create a pull secret."
echo
read -rp "  Create ghcr-pull-secret? [y/N]: " CREATE_PULL_SECRET
if [[ "${CREATE_PULL_SECRET,,}" == "y" ]]; then
  prompt_secret "GitHub username" GHCR_USER
  prompt_secret "GitHub PAT (read:packages scope)" GHCR_TOKEN
  for ns in ai-agent embedding ingestion; do
    if kubectl get secret ghcr-pull-secret -n "$ns" &>/dev/null; then
      warn "  ghcr-pull-secret already exists in $ns — skipping"
    else
      kubectl create secret docker-registry ghcr-pull-secret \
        --namespace "$ns" \
        --docker-server=ghcr.io \
        --docker-username="${GHCR_USER}" \
        --docker-password="${GHCR_TOKEN}"
      ok "  Created ghcr-pull-secret in namespace $ns"
    fi
  done
else
  info "Skipping ghcr-pull-secret — ensure packages are public or create the secret manually."
fi

# ── Deploy services ────────────────────────────────────────────────────────────
info "Deploying all services via Helm..."
bash "${REPO_ROOT}/deploy/install.sh"

# ── Done ───────────────────────────────────────────────────────────────────────
echo
echo "─────────────────────────────────────────────────────────────────"
ok "Bootstrap complete."
echo
echo "  Watch rollout:"
echo "    kubectl get pods -A -w"
echo
echo "  Re-deploy after editing a chart or values.yaml:"
echo "    ./deploy/install.sh"
echo
echo "  Service endpoints (available once pods reach Running):"
echo "    Open-WebUI   →  http://${NODE_IP}:30080"
echo "    vLLM API     →  http://${NODE_IP}:30000/v1"
echo "    ai-agent     →  http://${NODE_IP}:30081"
echo "    embedding    →  http://${NODE_IP}:30082"
echo "    ingestion    →  http://${NODE_IP}:30083"
echo "    Qdrant REST  →  http://${NODE_IP}:30333"
echo "─────────────────────────────────────────────────────────────────"
