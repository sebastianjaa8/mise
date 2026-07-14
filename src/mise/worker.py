"""Background worker: polls the job queue and processes catalog-refresh
jobs so an admin request never blocks on the actual (slow) rebuild work.
Run continuously with `python -m mise.worker`, or process a single queued
job and exit with `python -m mise.worker --once` (used by the smoke test and
by a one-shot cron-style invocation).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from mise.build_index import build as build_faiss_index
from mise.dataset import FeatureEncoder
from mise.job_queue import Job, JobQueue
from mise.model import TwoTowerModel


def _rebuild_index(job: Job):
    """Recompute item embeddings from the current catalog + trained model,
    then rebuild the FAISS index — the thing a real catalog-refresh job
    would do after new recipes are added, without retraining the towers."""
    recipes = pd.read_csv("data/recipes.csv")
    encoder = FeatureEncoder.build()
    model = TwoTowerModel(user_dim=encoder.user_dim, item_dim=encoder.item_dim)
    model.load_state_dict(torch.load("artifacts/model.pt"))
    model.eval()

    with torch.no_grad():
        item_embeddings = model.item_tower(torch.tensor(encoder.encode_items(recipes))).numpy().astype("float32")

    np.save("artifacts/item_embeddings.npy", item_embeddings)
    np.save("artifacts/item_ids.npy", recipes.recipe_id.to_numpy())
    build_faiss_index()
    return {"catalog_size": len(recipes)}


HANDLERS = {"rebuild_index": _rebuild_index}


def process_one(queue: JobQueue) -> bool:
    job = queue.claim_next()
    if job is None:
        return False
    handler = HANDLERS.get(job.job_type)
    if handler is None:
        queue.mark_failed(job.id, f"no handler for job_type={job.job_type}")
        return True
    try:
        result = handler(job)
        queue.mark_done(job.id, result)
        print(f"job {job.id} ({job.job_type}) done: {result}")
    except Exception as e:
        queue.mark_failed(job.id, str(e))
        print(f"job {job.id} ({job.job_type}) failed: {e}")
    return True


def run(poll_interval=2.0, once=False):
    queue = JobQueue()
    if once:
        process_one(queue)
        return
    print("worker polling for jobs...")
    while True:
        if not process_one(queue):
            time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="process a single pending job and exit")
    args = parser.parse_args()
    run(once=args.once)
