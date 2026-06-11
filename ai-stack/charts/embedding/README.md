# embedding

Serves `nomic-ai/nomic-embed-text-v1.5` as an HTTP embedding API on port 30082. Used by both the ingestion pipeline (to embed documents) and the AI agent (to embed queries at retrieval time).

## Key values

| Key | Default | Description |
|---|---|---|
| `image.tag` | `latest` | Image tag — built by CI from this repo |
| `model.name` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace embedding model to load |
| `service.nodePort` | `30082` | External port on each cluster node |
| `resources.limits.memory` | `24Gi` | Upper memory limit — reduce if sharing node RAM |
| `nodeSelector` | `<your-node-name>` | Pin to the node with sufficient RAM for the model |

## Standalone install

```bash
helm upgrade --install embedding ./ai-stack/charts/embedding \
  -n embedding --create-namespace \
  --set nodeSelector."kubernetes\\.io/hostname"=<your-node-name>
```
