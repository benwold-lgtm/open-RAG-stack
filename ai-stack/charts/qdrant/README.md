# qdrant

Qdrant vector database that stores embeddings and metadata for all ingested documents. Collections are created automatically by the ingestion service on first use.

## Key values

| Key | Default | Description |
|---|---|---|
| `image.tag` | `v1.9.0` | Qdrant image version (pinned) |
| `storage.size` | `50Gi` | PVC size for vector storage — scale up for large corpora |
| `storage.storageClassName` | `local-path` | StorageClass — must exist on your cluster |
| `service.nodePort` | `30333` | External REST port on each cluster node |
| `service.grpcNodePort` | `30334` | External gRPC port |
| `security.secretName` | `qdrant-secrets` | Kubernetes Secret holding `QDRANT_API_KEY` |
| `nodeSelector` | `<your-node-name>` | Pin to the node with the persistent storage |

## Standalone install

Requires the `qdrant-secrets` Kubernetes Secret (created by `scripts/bootstrap.sh`). See the top-level [README](../../../../README.md) for the full bootstrap process.

```bash
helm upgrade --install qdrant ./ai-stack/charts/qdrant \
  -n qdrant --create-namespace \
  --set nodeSelector."kubernetes\\.io/hostname"=<your-node-name>
```
