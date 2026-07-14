"""Gateway: the only service a client actually talks to. Composes the
candidate-gen and ranking services over real HTTP calls — this is what
"independently scalable services" means in practice: two network hops
instead of two in-process function calls, each service deployable and
scaled on its own.
"""
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

CANDIDATE_SERVICE_URL = os.environ.get("CANDIDATE_SERVICE_URL", "http://localhost:8001")
RANKING_SERVICE_URL = os.environ.get("RANKING_SERVICE_URL", "http://localhost:8002")

_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(timeout=5.0)
    yield
    await _client.aclose()


app = FastAPI(title="mise gateway", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # GET /metrics — request count/latency/status by route


@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            candidate_health = (await client.get(f"{CANDIDATE_SERVICE_URL}/health")).json()
            ranking_health = (await client.get(f"{RANKING_SERVICE_URL}/health")).json()
        except httpx.ConnectError as e:
            raise HTTPException(status_code=503, detail=f"downstream service unreachable: {e}")
    return {"status": "ok", "candidate_service": candidate_health, "ranking_service": ranking_health}


@app.get("/recommend")
async def recommend(user_id: int, k: int = 10, pool_size: int = 50):
    candidate_resp = await _client.get(f"{CANDIDATE_SERVICE_URL}/candidates", params={"user_id": user_id, "k": pool_size})
    if candidate_resp.status_code != 200:
        raise HTTPException(status_code=candidate_resp.status_code, detail=candidate_resp.json().get("detail"))
    candidates = candidate_resp.json()["candidates"]

    rerank_resp = await _client.post(
        f"{RANKING_SERVICE_URL}/rerank",
        json={"user_id": user_id, "candidates": candidates, "k": k},
    )
    if rerank_resp.status_code != 200:
        raise HTTPException(status_code=rerank_resp.status_code, detail=rerank_resp.json().get("detail"))
    return rerank_resp.json()
