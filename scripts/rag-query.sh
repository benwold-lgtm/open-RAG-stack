#!/usr/bin/env bash
set -euo pipefail
# Send a question to the AI agent and display the response.
# Usage:
#   ./rag-query.sh
#   ./rag-query.sh "What is an AI factory reference architecture?"

AI_AGENT_URL="${AI_AGENT_URL:-http://<your-gpu-node-ip>:30081}"
# Bearer credential for the (optionally) authenticated data plane; unset = send nothing.
SERVICE_TOKEN="${SERVICE_TOKEN:-}"
AUTH_HEADER=(); [[ -n "$SERVICE_TOKEN" ]] && AUTH_HEADER=(-H "Authorization: Bearer ${SERVICE_TOKEN}")

CYAN='\033[0;36m'; GREEN='\033[0;32m'; NC='\033[0m'

QUERY="$*"
if [[ -z "$QUERY" ]]; then
    read -rp "Question: " QUERY
fi
if [[ -z "$QUERY" ]]; then
    echo "Error: question is required." >&2; exit 1
fi

PAYLOAD=$(printf '{"model":"rag","messages":[{"role":"user","content":"%s"}]}' \
    "$(echo "$QUERY" | sed 's/"/\\"/g')")

echo ""
echo -e "${CYAN}Query:${NC} $QUERY"
echo -e "${GREEN}────────────────────────────────────────────────────────────────${NC}"

RESPONSE=$(curl -s -X POST "${AI_AGENT_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]}" \
    -d "$PAYLOAD")

echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
choices = d.get('choices', [])
if choices:
    print(choices[0]['message']['content'])
else:
    print(json.dumps(d, indent=2))
"
echo ""
