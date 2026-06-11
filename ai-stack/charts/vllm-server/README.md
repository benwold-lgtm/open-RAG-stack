# vllm-server

vLLM inference server that loads any HuggingFace-compatible model and exposes an OpenAI-compatible API on port 30000. Requires a GPU node with sufficient VRAM (RTX 3090 / 24 GB recommended for 7B–14B models).

## Key values

| Key | Default | Description |
|---|---|---|
| `image.tag` | `latest` | vLLM image tag (3rd-party; pin to a semver for production) |
| `model.name` | `""` | **Required.** HuggingFace model ID, e.g. `mistralai/Mistral-7B-Instruct-v0.3` |
| `server.maxModelLen` | `8192` | Maximum context window in tokens |
| `server.gpuMemoryUtilization` | `0.90` | Fraction of GPU VRAM vLLM may use |
| `server.maxNumSeqs` | `4` | Max concurrent request slots |
| `persistence.size` | `500Gi` | PVC size for model weight storage |
| `patches.enabled` | `false` | Enable Qwen3 Genesis memory patches (RTX 3090 only) |
| `nodeSelector` | `<your-node-name>` | Pin to the GPU node |

## Standalone install

Requires the `hf-token-secret` Kubernetes Secret (created by `scripts/bootstrap.sh`). See the top-level [README](../../../../README.md) for the full bootstrap process.

```bash
helm upgrade --install vllm-server ./ai-stack/charts/vllm-server \
  -n ai-stack --create-namespace \
  --set model.name="mistralai/Mistral-7B-Instruct-v0.3" \
  --set nodeSelector."kubernetes\\.io/hostname"=<your-node-name>
```
