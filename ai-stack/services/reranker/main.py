import os
import asyncio
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

app = FastAPI(title="Reranker Service")

rerank_semaphore = asyncio.Semaphore(1)

MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading reranker model: {MODEL_NAME} on {device}")
model = CrossEncoder(MODEL_NAME, device=device)
print(f"Reranker model loaded successfully on {device}")


class RerankRequest(BaseModel):
    query: str
    passages: list[str]
    top_n: int = 5

class RerankResult(BaseModel):
    index: int
    score: float
    text: str


@app.post("/rerank", response_model=list[RerankResult])
async def rerank(request: RerankRequest):
    if not request.passages:
        return []
    pairs = [(request.query, p) for p in request.passages]
    loop = asyncio.get_event_loop()
    async with rerank_semaphore:
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(pairs).tolist()
        )
    ranked = sorted(
        [{"index": i, "score": float(scores[i]), "text": request.passages[i]}
         for i in range(len(scores))],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:request.top_n]


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME, "device": device}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model"}]
    }
