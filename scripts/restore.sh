#!/usr/bin/env bash
set -euo pipefail
# Restore a backup created by backup.sh into the Docker Compose deployment.
#
#   - Qdrant: recovers each collection from its snapshot (replaces that collection).
#   - chat-ui / ingestion SQLite DBs: stops the service, copies the DB back, restarts.
#   - ingestion files: extracted back into the volume.
#
# WARNING: this OVERWRITES current data. Make a fresh backup first if in doubt.
#
# Usage:  ./scripts/restore.sh <backup_dir>
# Env:    QDRANT_URL (default http://localhost:6333), QDRANT_API_KEY (default empty)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DIR="${1:?usage: restore.sh <backup_dir>}"
[ -d "$DIR" ] || { echo "No such backup dir: $DIR" >&2; exit 1; }
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
info() { echo -e "${CYAN}[..]${NC}   $*"; }
warn() { echo -e "${YELLOW}[!!]${NC}   $*"; }

dc()    { docker compose "$@"; }
qcurl() { curl -fsS ${QDRANT_API_KEY:+-H "api-key: ${QDRANT_API_KEY}"} "$@"; }
urlenc(){ python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }

echo
warn "Restoring from ${DIR} — this overwrites current data. Ctrl-C within 5s to abort."
sleep 5

# ── SQLite: stop the writer, copy the DB into the volume, restart ──
restore_sqlite() { # <service> <host-file> <db-path-in-container>
  if [ -f "$2" ]; then
    dc stop "$1" >/dev/null
    dc cp "$2" "$1:$3"
    dc start "$1" >/dev/null
    ok "$1 DB restored"
  else
    warn "$1: $(basename "$2") not in backup — skipped"
  fi
}
restore_sqlite chat-ui   "${DIR}/chat_ui.db"        /data/chat_ui.db
restore_sqlite ingestion "${DIR}/ingestion.db"      /app/data/ingestion.db

# ── ingestion files ──
if [ -f "${DIR}/ingestion-files.tar.gz" ]; then
  info "ingestion files..."
  dc exec -T ingestion sh -c 'rm -rf /app/data/files && tar xzf - -C /app/data' < "${DIR}/ingestion-files.tar.gz"
  ok "ingestion files restored"
fi

# ── Qdrant: recover each collection from its snapshot (multipart upload) ──
shopt -s nullglob
for f in "${DIR}"/qdrant/*.snapshot; do
  c="$(basename "$f" .snapshot)"
  enc="$(urlenc "$c")"
  info "qdrant: recovering ${c}..."
  qcurl -X POST "${QDRANT_URL}/collections/${enc}/snapshots/upload?priority=snapshot" \
    -H "Content-Type:multipart/form-data" -F "snapshot=@${f}" >/dev/null
  ok "qdrant: ${c} restored"
done

echo
ok "Restore complete."
