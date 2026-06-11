# ingestion

Document ingestion pipeline that chunks, embeds, and stores documents in Qdrant. Supports URL scraping, PDF/text file upload via HTTP, and an optional watch folder for auto-ingestion.

## Key values

| Key | Default | Description |
|---|---|---|
| `image.tag` | `latest` | Image tag — built by CI from this repo |
| `qdrant.url` | cluster DNS | Qdrant REST endpoint |
| `embedding.url` | cluster DNS | Embedding service endpoint |
| `config.chunkSize` | `256` | Token chunk size for document splitting |
| `config.chunkOverlap` | `25` | Token overlap between consecutive chunks |
| `config.embeddingDim` | `768` | Must match the embedding model's output dimension |
| `storage.size` | `5Gi` | PVC size for ingested document storage |
| `watchDir.enabled` | `false` | Enable auto-ingestion from a host directory |
| `watchDir.hostPath` | `""` | Absolute path on the node to watch (required if enabled) |
| `watchDir.pollInterval` | `60` | Seconds between directory polls |
| `nodeSelector` | `<your-node-name>` | Pin to the node where the watch directory lives |

## Standalone install

Requires the `qdrant-secrets` Kubernetes Secret (created by `scripts/bootstrap.sh`). See the top-level [README](../../../../README.md) for the full bootstrap process.

```bash
helm upgrade --install ingestion ./ai-stack/charts/ingestion \
  -n ingestion --create-namespace \
  --set nodeSelector."kubernetes\\.io/hostname"=<your-node-name>
```
