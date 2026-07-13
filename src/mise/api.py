"""FastAPI serving layer for the recipe retrieval model.

GET /recommend?user_id=0&k=10        -> ranked recipes for a known user
GET /recommend?equipment=...&k=10    -> cold-start ranking from raw profile features
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from mise.retrieve import Recommender

_recommender: Optional[Recommender] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _recommender
    _recommender = Recommender()
    yield


app = FastAPI(title="mise retrieval service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": int(_recommender.index.ntotal) if _recommender else 0}


@app.get("/recommend")
def recommend(
    user_id: Optional[int] = None,
    equipment: Optional[str] = Query(None, description="pipe-separated, e.g. blender|multi_cooker"),
    cuisine_affinity: Optional[str] = Query(None, description="pipe-separated, e.g. mexican|indian"),
    protein_target: Optional[float] = None,
    max_prep_min: Optional[float] = None,
    k: int = 10,
):
    if user_id is not None:
        try:
            return {"user_id": user_id, "results": _recommender.recommend_for_user(user_id, k)}
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    if equipment and protein_target is not None and max_prep_min is not None:
        results = _recommender.recommend_for_profile(
            equipment=set(equipment.split("|")),
            cuisine_affinity=set(cuisine_affinity.split("|")) if cuisine_affinity else set(),
            protein_target=protein_target,
            max_prep_min=max_prep_min,
            k=k,
        )
        return {"results": results}

    raise HTTPException(
        status_code=400,
        detail="pass either user_id, or equipment+protein_target+max_prep_min for a cold-start query",
    )
