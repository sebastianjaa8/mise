"""Feature encoding + torch Dataset for the two-tower retrieval model.

Note: user `persona` is kept in users.csv purely for eval/analysis
segmentation. It is deliberately NOT fed into the user tower — a real new
user won't arrive pre-labeled with a synthetic persona, so the tower has to
earn its predictions from equipment / protein target / prep budget / cuisine
affinity, the same signals a real onboarding form would collect.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mise.config import CUISINES, DIET_TAGS, EQUIPMENT


def _multi_hot(value: str, vocab: list[str]) -> np.ndarray:
    tags = set(value.split("|")) if isinstance(value, str) and value else set()
    return np.array([1.0 if v in tags else 0.0 for v in vocab], dtype=np.float32)


def _one_hot(value: str, vocab: list[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    return vec


@dataclass
class FeatureEncoder:
    item_dim: int
    user_dim: int

    @staticmethod
    def build():
        item_dim = len(CUISINES) + len(EQUIPMENT) + len(DIET_TAGS) + 3
        user_dim = len(EQUIPMENT) + len(CUISINES) + 2
        return FeatureEncoder(item_dim=item_dim, user_dim=user_dim)

    def encode_items(self, recipes_df: pd.DataFrame) -> np.ndarray:
        feats = []
        for row in recipes_df.itertuples(index=False):
            cuisine_vec = _one_hot(row.cuisine, CUISINES)
            equipment_vec = _multi_hot(row.equipment, EQUIPMENT)
            diet_vec = _multi_hot(row.diet_tags, DIET_TAGS)
            scalar = np.array([
                row.protein_g / 100.0,
                row.prep_time_min / 240.0,
                row.pop_bias,
            ], dtype=np.float32)
            feats.append(np.concatenate([cuisine_vec, equipment_vec, diet_vec, scalar]))
        return np.stack(feats).astype(np.float32)

    def encode_users(self, users_df: pd.DataFrame) -> np.ndarray:
        feats = []
        for row in users_df.itertuples(index=False):
            equipment_vec = _multi_hot(row.equipment, EQUIPMENT)
            cuisine_vec = _multi_hot(row.cuisine_affinity, CUISINES)
            scalar = np.array([
                row.protein_target / 100.0,
                row.max_prep_min / 240.0,
            ], dtype=np.float32)
            feats.append(np.concatenate([equipment_vec, cuisine_vec, scalar]))
        return np.stack(feats).astype(np.float32)

    def encode_single_user(self, equipment: set[str], cuisine_affinity: set[str],
                            protein_target: float, max_prep_min: float) -> np.ndarray:
        equipment_vec = _multi_hot("|".join(equipment), EQUIPMENT)
        cuisine_vec = _multi_hot("|".join(cuisine_affinity), CUISINES)
        scalar = np.array([protein_target / 100.0, max_prep_min / 240.0], dtype=np.float32)
        return np.concatenate([equipment_vec, cuisine_vec, scalar]).astype(np.float32)


class PairDataset(Dataset):
    """(user_row_idx, positive_item_row_idx) pairs for in-batch-negative training."""

    def __init__(self, interactions_df: pd.DataFrame, user_id_to_row: dict, item_id_to_row: dict):
        positives = interactions_df[interactions_df.is_positive == 1]
        self.user_rows = positives.user_id.map(user_id_to_row).to_numpy()
        self.item_rows = positives.recipe_id.map(item_id_to_row).to_numpy()

    def __len__(self):
        return len(self.user_rows)

    def __getitem__(self, idx):
        return torch.tensor(self.user_rows[idx]), torch.tensor(self.item_rows[idx])
