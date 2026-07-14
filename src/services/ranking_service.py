"""Ranking service: takes a candidate pool + two-tower scores from the
candidate-gen service and reorders it with the LightGBM ranker. Runs
independently so it can scale on its own cost profile (a GBDT pass per
candidate, heavier per-request than a single FAISS lookup) and could be
swapped for a different ranking model without touching retrieval at all.
"""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from mise.rerank import Reranker

_state: dict = {}


class Candidate(BaseModel):
    recipe_id: int
    two_tower_score: float


class RerankRequest(BaseModel):
    user_id: int
    candidates: list[Candidate]
    k: int = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["users"] = pd.read_csv("data/users.csv")
    _state["recipes"] = pd.read_csv("data/recipes.csv")
    _state["reranker"] = Reranker()
    yield
    _state.clear()


app = FastAPI(title="mise ranking service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "ranker_loaded": "reranker" in _state}


@app.post("/rerank")
def rerank(req: RerankRequest):
    two_tower_scores = {(req.user_id, c.recipe_id): c.two_tower_score for c in req.candidates}
    candidate_ids = [c.recipe_id for c in req.candidates]

    ranked_ids = _state["reranker"].rerank(
        req.user_id, candidate_ids, two_tower_scores, _state["users"], _state["recipes"], req.k
    )
    recipes = _state["recipes"].set_index("recipe_id").loc[ranked_ids]
    return {
        "user_id": req.user_id,
        "results": [
            {"recipe_id": int(rid), "title": row.title, "cuisine": row.cuisine,
             "protein_g": int(row.protein_g), "prep_time_min": int(row.prep_time_min)}
            for rid, (_, row) in zip(ranked_ids, recipes.iterrows())
        ],
    }
