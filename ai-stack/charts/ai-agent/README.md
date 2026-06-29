# ai-agent

FastAPI RAG agent that receives queries from the chat UI, retrieves relevant context from Qdrant, and calls vLLM to generate responses. This is the orchestration layer between the chat UI and the rest of the stack.

## Key values

| Key | Default | Description |
|---|---|---|
| `image.tag` | `latest` | Image tag — built by CI from this repo |
| `vllm.baseUrl` | `""` | vLLM base URL, e.g. `http://<NODE_IP>:30000/v1` |
| `vllm.model` | `""` | HuggingFace model ID loaded by vllm-server |
| `qdrant.url` | cluster DNS | Qdrant REST endpoint (default uses in-cluster DNS) |
| `embedding.url` | cluster DNS | Embedding service endpoint (default uses in-cluster DNS) |
| `secret.name` | `ai-agent-secrets` | Kubernetes Secret holding `QDRANT_API_KEY` and `BRAVE_API_KEY` |
| `nodeSelector` | `<your-node-name>` | Pin to the GPU node running the rest of the stack |

## Standalone install

Requires the `ai-agent-secrets` Kubernetes Secret (created by `scripts/bootstrap.sh`). See the top-level [README](../../../../README.md) for the full bootstrap process.

```bash
helm upgrade --install ai-agent ./ai-stack/charts/ai-agent \
  -n ai-agent --create-namespace \
  --set vllm.baseUrl="http://<NODE_IP>:30000/v1" \
  --set vllm.model="mistralai/Mistral-7B-Instruct-v0.3" \
  --set nodeSelector."kubernetes\\.io/hostname"=<your-node-name>
```
