"""Candidate-generation service: two-tower + FAISS retrieval only.

This is the "broad and cheap" half of the pipeline, split out so it can be
scaled independently of ranking — retrieval is a single FAISS lookup per
request; ranking runs a GBDT over every candidate, which is a different
(heavier) cost profile per request. Owns nothing the ranking service needs
to know about beyond a candidate id + its own similarity score.
"""
from contextlib import asynccontextmanager
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from mise.dataset import FeatureEncoder
from mise.job_queue import JobQueue
from mise.model import TwoTowerModel

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    users = pd.read_csv("data/users.csv")
    encoder = FeatureEncoder.build()
    model = TwoTowerModel(user_dim=encoder.user_dim, item_dim=encoder.item_dim)
    model.load_state_dict(torch.load("artifacts/model.pt"))
    model.eval()

    _state["users"] = users
    _state["encoder"] = encoder
    _state["model"] = model
    _state["index"] = faiss.read_index("artifacts/items.faiss")
    _state["item_ids"] = np.load("artifacts/item_ids.npy")
    _state["queue"] = JobQueue()
    yield
    _state.clear()


app = FastAPI(title="mise candidate-gen service", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # GET /metrics


@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": int(_state["index"].ntotal) if "index" in _state else 0}


@app.post("/admin/refresh-catalog")
def refresh_catalog():
    """Enqueue an index rebuild instead of doing it inline — the request
    returns immediately, the actual (slow) recompute happens in the worker
    process. See `mise.worker` / `python -m mise.worker`."""
    job_id = _state["queue"].enqueue("rebuild_index")
    return {"job_id": job_id, "status": "queued"}


@app.get("/admin/jobs/{job_id}")
def job_status(job_id: int):
    job = _state["queue"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
    return job


@app.post("/admin/reload-index")
def reload_index():
    """Swap in whatever the worker last wrote to disk, without restarting
    this process — the same "hot swap a new index version" pattern a real
    deployment would use instead of a service bounce per catalog refresh."""
    _state["index"] = faiss.read_index("artifacts/items.faiss")
    _state["item_ids"] = np.load("artifacts/item_ids.npy")
    return {"status": "reloaded", "catalog_size": int(_state["index"].ntotal)}


@app.get("/candidates")
def candidates(user_id: int, k: int = 50):
    users = _state["users"]
    user_row = users[users.user_id == user_id]
    if user_row.empty:
        raise HTTPException(status_code=404, detail=f"unknown user_id: {user_id}")

    user_vec = _state["encoder"].encode_users(user_row)[0]
    with torch.no_grad():
        user_emb = _state["model"].user_tower(torch.tensor(user_vec).unsqueeze(0)).squeeze(0).numpy()

    scores, idx = _state["index"].search(user_emb.reshape(1, -1).astype("float32"), k)
    recipe_ids = [int(r) for r in _state["item_ids"][idx[0]]]
    return {
        "user_id": user_id,
        "candidates": [{"recipe_id": rid, "two_tower_score": float(s)} for rid, s in zip(recipe_ids, scores[0])],
    }
