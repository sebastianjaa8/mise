"""Query-time retrieval: encode a user (existing or brand new) with the user
tower, then hit the FAISS index for the nearest item embeddings.
"""
import faiss
import numpy as np
import pandas as pd
import torch

from mise.dataset import FeatureEncoder
from mise.model import TwoTowerModel


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

    def _search(self, user_vec: np.ndarray, k: int):
        query = user_vec.reshape(1, -1).astype("float32")
        scores, idx = self.index.search(query, k)
        recipe_ids = self.item_ids[idx[0]]
        rows = self.recipes.set_index("recipe_id").loc[recipe_ids]
        return [
            {"recipe_id": int(rid), "title": row.title, "cuisine": row.cuisine,
             "protein_g": int(row.protein_g), "prep_time_min": int(row.prep_time_min),
             "score": float(score)}
            for rid, (_, row), score in zip(recipe_ids, rows.iterrows(), scores[0])
        ]

    def recommend_for_user(self, user_id: int, k: int = 10):
        user_row = self.users[self.users.user_id == user_id]
        if user_row.empty:
            raise KeyError(f"unknown user_id: {user_id}")
        user_vec = self.encoder.encode_users(user_row)[0]
        with torch.no_grad():
            user_emb = self.model.user_tower(torch.tensor(user_vec).unsqueeze(0)).squeeze(0).numpy()
        return self._search(user_emb, k)

    def recommend_for_profile(self, equipment: set[str], cuisine_affinity: set[str],
                               protein_target: float, max_prep_min: float, k: int = 10):
        """Cold-start path: a brand new user with no interaction history yet
        — still gets a real personalized ranking from onboarding-form
        features alone, since the user tower never depended on history."""
        user_vec = self.encoder.encode_single_user(equipment, cuisine_affinity, protein_target, max_prep_min)
        with torch.no_grad():
            user_emb = self.model.user_tower(torch.tensor(user_vec).unsqueeze(0)).squeeze(0).numpy()
        return self._search(user_emb, k)
