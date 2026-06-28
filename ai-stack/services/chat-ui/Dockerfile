# NOTE: build context must be the REPO ROOT (not this dir) so the shared rag_auth
# module under ai-stack/lib/ is reachable. docker-compose sets:
#   build: { context: ., dockerfile: ai-stack/services/chat-ui/Dockerfile }
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Shared auth module first (changes rarely → better layer caching).
COPY ai-stack/lib/rag_auth /tmp/rag_auth
RUN pip install --no-cache-dir /tmp/rag_auth

COPY ai-stack/services/chat-ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai-stack/services/chat-ui/main.py .

# Non-root; /data holds the SQLite DB (mount a volume here).
RUN useradd -m -u 1000 appuser && mkdir -p /data && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8006

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]
