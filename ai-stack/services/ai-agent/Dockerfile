# NOTE: build context must be the REPO ROOT (not this dir) so the shared rag_auth
# module under ai-stack/lib/ is reachable. docker-compose sets:
#   build: { context: ., dockerfile: ai-stack/services/ai-agent/Dockerfile }
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Shared auth module first (changes rarely → better layer caching).
COPY ai-stack/lib/rag_auth /tmp/rag_auth
RUN pip install --no-cache-dir /tmp/rag_auth

# Install dependencies first (better layer caching)
COPY ai-stack/services/ai-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ai-stack/services/ai-agent/main.py .

# Non-root user for security (production best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Start the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
