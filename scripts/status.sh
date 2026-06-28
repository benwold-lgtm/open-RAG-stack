#!/usr/bin/env bash
# Quick overview of all AI stack pods across namespaces.
NAMESPACES=(ai-stack ai-agent embedding ingestion qdrant chat-ui)

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║           open-RAG-stack — Pod Status               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

for ns in "${NAMESPACES[@]}"; do
    echo "── $ns ──────────────────────────────────────────────"
    kubectl get pods -n "$ns" --no-headers \
        -o custom-columns="NAME:.metadata.name,READY:.status.containerStatuses[0].ready,STATUS:.status.phase,RESTARTS:.status.containerStatuses[0].restartCount,NODE:.spec.nodeName" \
        2>/dev/null || echo "  (no pods)"
    echo ""
done
