#!/usr/bin/env bash
set -euo pipefail
# Back up the open-RAG-stack data for the Docker Compose deployment:
#   - Qdrant vector collections (consistent snapshots via the snapshot API)
#   - chat-ui SQLite DB   (users, conversations, messages, sessions)
#   - ingestion SQLite DB (the FTS index + documents table) and uploaded files
#
# Takes an ONLINE, consistent copy of each store, so you do NOT need to stop the stack.
# (SQLite is single-writer / single-replica — see the README "Backup & restore" notes.)
#
# Does NOT back up secrets / .env — manage those separately and securely.
#
# Usage:  ./scripts/backup.sh [output_dir]
#         (default output: ./backups/<UTC timestamp>)
# Env:    QDRANT_URL (default http://localhost:6333), QDRANT_API_KEY (default empty)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
info() { echo -e "${CYAN}[..]${NC}   $*"; }

TS="$(date -u +%Y%m%d-%H%M%SZ)"
OUT="${1:-./backups/$TS}"
mkdir -p "$OUT/qdrant"

dc()    { docker compose "$@"; }
qcurl() { curl -fsS ${QDRANT_API_KEY:+-H "api-key: ${QDRANT_API_KEY}"} "$@"; }
urlenc(){ python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }

echo
info "Backing up to ${OUT}"

# ── Qdrant: one snapshot per collection, downloaded, then the server copy removed ──
info "Qdrant collections..."
collections="$(qcurl "${QDRANT_URL}/collections" | python3 -c 'import sys,json; [print(c["name"]) for c in json.load(sys.stdin)["result"]["collections"]]')"
while IFS= read -r c; do
  [ -z "$c" ] && continue
  enc="$(urlenc "$c")"
  snap="$(qcurl -X POST "${QDRANT_URL}/collections/${enc}/snapshots" | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["name"])')"
  snapenc="$(urlenc "$snap")"   # snapshot name embeds the collection name, so it can contain spaces too
  qcurl "${QDRANT_URL}/collections/${enc}/snapshots/${snapenc}" -o "${OUT}/qdrant/${c}.snapshot"
  qcurl -X DELETE "${QDRANT_URL}/collections/${enc}/snapshots/${snapenc}" >/dev/null
  ok "qdrant: ${c}"
done <<< "$collections"

# ── SQLite: online backup (sqlite3 .backup) inside the container, streamed to host ──
sqlite_backup() { # <service> <db-path-in-container> <out-file>
  dc exec -T "$1" python -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect('/tmp/_bk.db'); s.backup(d); d.close(); s.close()" "$2"
  dc exec -T "$1" cat /tmp/_bk.db > "$3"
  dc exec -T "$1" rm -f /tmp/_bk.db
  ok "$1 DB -> $(basename "$3")"
}
info "chat-ui DB..."  ; sqlite_backup chat-ui   /data/chat_ui.db        "${OUT}/chat_ui.db"
info "ingestion DB..."; sqlite_backup ingestion /app/data/ingestion.db  "${OUT}/ingestion.db"

# ── ingestion uploaded files (page images, stored docs) ──
info "ingestion files..."
dc exec -T ingestion tar czf - -C /app/data files > "${OUT}/ingestion-files.tar.gz"
ok "ingestion files -> ingestion-files.tar.gz"

echo
ok "Backup complete: ${OUT}"
du -sh "$OUT" 2>/dev/null | sed 's/^/       total: /'
