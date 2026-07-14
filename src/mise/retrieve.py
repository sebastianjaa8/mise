"""Query-time retrieval: encode a user (existing or brand new) with the user
tower, hit the FAISS index for a broad candidate pool, then hand that pool to
the ranking stage (if trained) for the final reorder. Falls back to
retrieval-only ordering when no ranker artifact exists yet.
"""
import os

import faiss
import numpy as np
import pandas as pd
import torch

from mise.dataset import FeatureEncoder
from mise.model import TwoTowerModel

CANDIDATE_POOL_SIZE = 50


class Recommender:
    def __init__(self, artifacts_dir="artifacts", data_dir="data"):
        self.recipes = pd.read_csv(f"{data_dir}/recipes.csv")
        self.users = pd.read_csv(f"{data_dir}/users.csv")
        self.encoder = FeatureEncoder.build()

        self.model = TwoTowerModel(user_dim=self.encoder.user_dim, item_dim=self.encoder.item_dim)
        self.model.load_state_dict(torch.load(f"{artifacts_dir}/model.pt"))
        self.model.eval()

        self.index = faiss.read_index(f"{artifacts_dir}/items.faiss")
        self.item_ids = np.load(f"{artifacts_dir}/item_ids.npy")

        ranker_path = f"{artifacts_dir}/ranker.txt"
        self.reranker = None
        if os.path.exists(ranker_path):
            from mise.rerank import Reranker
            self.reranker = Reranker(ranker_path)

    def _search_candidates(self, user_vec: np.ndarray, pool_size: int):
        query = user_vec.reshape(1, -1).astype("float32")
        scores, idx = self.index.search(query, pool_size)
        recipe_ids = [int(r) for r in self.item_ids[idx[0]]]
        return recipe_ids, scores[0].tolist()

    def _format(self, recipe_ids: list) -> list:
        rows = self.recipes.set_index("recipe_id").loc[recipe_ids]
        return [
            {"recipe_id": int(rid), "title": row.title, "cuisine": row.cuisine,
             "protein_g": int(row.protein_g), "prep_time_min": int(row.prep_time_min)}
            for rid, (_, row) in zip(recipe_ids, rows.iterrows())
        ]

    def recommend_for_user(self, user_id: int, k: int = 10):
        user_row = self.users[self.users.user_id == user_id]
        if user_row.empty:
            raise KeyError(f"unknown user_id: {user_id}")
        user_vec = self.encoder.encode_users(user_row)[0]
        with torch.no_grad():
            user_emb = self.model.user_tower(torch.tensor(user_vec).unsqueeze(0)).squeeze(0).numpy()

        pool_size = CANDIDATE_POOL_SIZE if self.reranker else k
        candidate_ids, candidate_scores = self._search_candidates(user_emb, pool_size)

        if self.reranker:
            two_tower_scores = {(user_id, rid): s for rid, s in zip(candidate_ids, candidate_scores)}
            final_ids = self.reranker.rerank(user_id, candidate_ids, two_tower_scores, self.users, self.recipes, k)
        else:
            final_ids = candidate_ids[:k]
        return self._format(final_ids)

    def recommend_for_profile(self, equipment: set[str], cuisine_affinity: set[str],
                               protein_target: float, max_prep_min: float, k: int = 10):
        """Cold-start path: a brand new user with no interaction history yet
        — still gets a real personalized ranking from onboarding-form
        features alone, since the user tower never depended on history.
        Retrieval-only (no rerank): the ranker's features are keyed off a
        users.csv row, which a not-yet-onboarded profile doesn't have."""
        user_vec = self.encoder.encode_single_user(equipment, cuisine_affinity, protein_target, max_prep_min)
        with torch.no_grad():
            user_emb = self.model.user_tower(torch.tensor(user_vec).unsqueeze(0)).squeeze(0).numpy()
        candidate_ids, _ = self._search_candidates(user_emb, k)
        return self._format(candidate_ids)
