# open-webui

Open-WebUI chat interface pre-wired to route all queries through the AI agent service. Authentication is enabled by default — the first user to register becomes the admin.

## Key values

| Key | Default | Description |
|---|---|---|
| `image.tag` | `main` | Open-WebUI image tag (3rd-party; pin to a semver for production) |
| `vllm.baseUrl` | cluster DNS | AI agent endpoint — change only if routing via NodePort instead of cluster DNS |
| `auth.enabled` | `true` | Disable only on a fully trusted, isolated network |
| `storage.size` | `5Gi` | PVC size for chat history and user data |
| `service.nodePort` | `30080` | External port on each cluster node |
| `nodeSelector` | `<your-node-name>` | Pin to the target node |

## Standalone install

```bash
helm upgrade --install open-webui ./ai-stack/charts/open-webui \
  -n open-webui --create-namespace \
  --set nodeSelector."kubernetes\\.io/hostname"=<your-node-name>
```
