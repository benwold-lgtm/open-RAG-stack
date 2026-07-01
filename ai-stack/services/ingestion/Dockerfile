# NOTE: build context must be the REPO ROOT (not this dir) so the shared rag_auth
# module under ai-stack/lib/ is reachable. docker-compose sets:
#   build: { context: ., dockerfile: ai-stack/services/ingestion/Dockerfile }
FROM python:3.11-slim

WORKDIR /app

# Playwright browser binaries land here — shared, accessible to non-root user
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Shared auth module first (changes rarely → better layer caching).
COPY ai-stack/lib/rag_auth /tmp/rag_auth
RUN pip install --no-cache-dir /tmp/rag_auth

COPY ai-stack/services/ingestion/requirements.txt .
# libreoffice-writer/impress (headless) convert DOCX/PPTX -> PDF so office docs flow through the
# same page-aware pipeline as PDFs (page images, captions, OCR). --no-install-recommends keeps the
# image lean; PDF export does not need the Java runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libreoffice-writer \
    libreoffice-impress \
    && pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium \
    && chmod -R 755 /ms-playwright \
    && rm -rf /var/lib/apt/lists/*

COPY ai-stack/services/ingestion/main.py .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
