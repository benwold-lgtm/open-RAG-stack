#!/usr/bin/env bash
# Download all models required by open-RAG-stack to a local or NFS path.
# Run this ONCE on a machine with internet access (e.g. bengpu1 with NFS mounted).
# After this script completes, the stack runs with zero external network calls.
#
# Usage:
#   ./scripts/download-models.sh                        # uses default MODEL_DIR
#   MODEL_DIR=/mnt/nfs/models ./scripts/download-models.sh
#   HF_TOKEN=hf_xxx ./scripts/download-models.sh       # required for gated models (e.g. Llama)
#
# Prerequisites:
#   - Python 3.10+ with pip
#   - Sufficient disk space (~20 GB for all non-LLM models; LLM varies by choice)
#   - NFS mounted at MODEL_DIR if using shared storage
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()  { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }
skip() { echo -e "${YELLOW}[SKIP]${NC}  $*"; }

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-/mnt/nfs/models}"   # NFS mount point or local path
HF_TOKEN="${HF_TOKEN:-}"                    # Required only for gated HF models (Llama, etc.)

# Models downloaded by this script
EMBED_MODEL="nomic-ai/nomic-embed-text-v1.5"
RERANK_MODEL="BAAI/bge-reranker-v2-m3"
# LLM model is set separately — see bottom of this script

echo
echo "╔══════════════════════════════════════════════════╗"
echo "║     open-RAG-stack  —  Model Pre-Download        ║"
echo "╚══════════════════════════════════════════════════╝"
echo
info "Target directory : ${MODEL_DIR}"
info "HF cache (HF_HOME) will be set to: ${MODEL_DIR}/huggingface"
echo

# ── Verify target directory ────────────────────────────────────────────────────
if [[ ! -d "$MODEL_DIR" ]]; then
    die "MODEL_DIR '${MODEL_DIR}' does not exist. Mount NFS first or set MODEL_DIR to an existing path."
fi
if [[ ! -w "$MODEL_DIR" ]]; then
    die "MODEL_DIR '${MODEL_DIR}' is not writable by the current user ($(whoami))."
fi
ok "Model directory is writable"

# ── Set HF_HOME so all HuggingFace downloads land in MODEL_DIR ────────────────
export HF_HOME="${MODEL_DIR}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

# ── Ensure huggingface-hub CLI is available ────────────────────────────────────
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    info "Installing huggingface-hub..."
    pip install -q "huggingface-hub>=0.24.0"
fi
if ! python3 -c "import sentence_transformers" 2>/dev/null; then
    info "Installing sentence-transformers (needed to verify model loads)..."
    pip install -q "sentence-transformers>=3.0.0"
fi

HF_CLI="python3 -m huggingface_hub.commands.huggingface_cli"

# ── Helper: download a HF repo if not already cached ─────────────────────────
download_hf_model() {
    local repo_id="$1"
    local local_name="${repo_id//\//__}"   # nomic-ai/nomic-embed-text-v1.5 → nomic-ai__nomic-embed-text-v1.5
    local cache_marker="${HF_HOME}/hub/models--${local_name}"

    if [[ -d "$cache_marker" ]]; then
        skip "${repo_id} already cached — skipping download"
        return 0
    fi

    info "Downloading ${repo_id} ..."
    local token_arg=""
    [[ -n "$HF_TOKEN" ]] && token_arg="--token ${HF_TOKEN}"

    python3 -m huggingface_hub.commands.huggingface_cli download \
        ${token_arg} \
        --cache-dir "${HF_HOME}/hub" \
        "${repo_id}" || die "Failed to download ${repo_id}"

    ok "${repo_id} downloaded"
}

# ── 1. Embedding model ─────────────────────────────────────────────────────────
echo
echo "── Embedding Model ──────────────────────────────────"
download_hf_model "${EMBED_MODEL}"

# Verify it loads
info "Verifying ${EMBED_MODEL} loads correctly..."
python3 - <<EOF
import os
os.environ["HF_HOME"] = "${HF_HOME}"
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("${EMBED_MODEL}", trust_remote_code=True)
v = m.encode(["test"])
assert v.shape[1] == 768, f"Expected 768 dims, got {v.shape[1]}"
print("  Embedding model OK — 768 dims confirmed")
EOF
ok "Embedding model verified"

# ── 2. Reranker model ──────────────────────────────────────────────────────────
echo
echo "── Reranker Model ───────────────────────────────────"
download_hf_model "${RERANK_MODEL}"

# Verify it loads
info "Verifying ${RERANK_MODEL} loads correctly..."
python3 - <<EOF
import os
os.environ["HF_HOME"] = "${HF_HOME}"
from sentence_transformers import CrossEncoder
m = CrossEncoder("${RERANK_MODEL}")
scores = m.predict([["what is RAG?", "RAG stands for Retrieval Augmented Generation"]])
assert len(scores) == 1
print(f"  Reranker model OK — test score: {scores[0]:.4f}")
EOF
ok "Reranker model verified"

# ── 3. Docling layout models ───────────────────────────────────────────────────
echo
echo "── Docling Document Parsing Models ──────────────────"
DOCLING_ARTIFACTS="${MODEL_DIR}/docling"
mkdir -p "${DOCLING_ARTIFACTS}"

if [[ -d "${DOCLING_ARTIFACTS}/TableFormer" ]] && [[ -d "${DOCLING_ARTIFACTS}/layout" ]]; then
    skip "Docling models already cached — skipping download"
else
    if ! python3 -c "import docling" 2>/dev/null; then
        info "Installing docling for model download..."
        pip install -q "docling>=2.0.0"
    fi

    info "Downloading Docling layout + TableFormer models..."
    DOCLING_ARTIFACTS_PATH="${DOCLING_ARTIFACTS}" python3 - <<EOF
import os
os.environ["DOCLING_ARTIFACTS_PATH"] = "${DOCLING_ARTIFACTS}"
from docling.document_converter import DocumentConverter
# Instantiating the converter triggers model downloads
DocumentConverter()
print("  Docling models downloaded")
EOF
    ok "Docling models downloaded to ${DOCLING_ARTIFACTS}"
fi

# ── 4. LLM model ───────────────────────────────────────────────────────────────
echo
echo "── LLM Model ────────────────────────────────────────"
warn "LLM download is skipped — the production LLM has not been selected yet."
warn "Once chosen, download it with:"
echo
echo "    HF_TOKEN=hf_xxx \\"
echo "    HF_HOME=${HF_HOME} \\"
echo "    huggingface-cli download <model-id> \\"
echo "        --local-dir ${MODEL_DIR}/huggingface/hub/<model-id>"
echo
warn "Then set model.name in ai-stack/charts/vllm-server/values.yaml"
warn "and update the vLLM Helm chart to use MODEL_DIR instead of downloading at pod startup."
warn "See docs/ENHANCEMENT-PLAN.md Phase 1.2 for Helm chart changes."

# ── Summary ────────────────────────────────────────────────────────────────────
echo
echo "╔══════════════════════════════════════════════════╗"
echo "║                    Summary                       ║"
echo "╚══════════════════════════════════════════════════╝"
ok  "Embedding model  : ${EMBED_MODEL}"
ok  "Reranker model   : ${RERANK_MODEL}"
ok  "Docling models   : ${DOCLING_ARTIFACTS}"
warn "LLM              : pending model selection"
echo
info "All models are stored under: ${MODEL_DIR}"
info "Set HF_HOME=${HF_HOME} in all service deployments (Phase 1.2)"
info "Set DOCLING_ARTIFACTS_PATH=${DOCLING_ARTIFACTS} in ingestion deployment (Phase 4.2)"
echo
ok "Phase 1.1 complete."
