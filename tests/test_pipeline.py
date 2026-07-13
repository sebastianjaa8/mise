"""End-to-end smoke test: generate a small dataset, train briefly, build the
index, and check retrieval actually beats a popularity baseline. This is the
one check that fails if the retrieval pipeline breaks — not a unit test per
function, just "does the whole thing still work."
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mise.data_gen import make_interactions, make_recipes, make_users
from mise.dataset import FeatureEncoder, PairDataset
from mise.evaluate import popularity_ranking, true_relevant_sets, two_tower_rankings
from mise.metrics import recall_at_k
from mise.model import TwoTowerModel
from mise.train import split_positive_interactions


def test_retrieval_beats_popularity_baseline():
    torch.manual_seed(0)

    recipes = make_recipes(n=150, seed=1)
    users = make_users(n=80, seed=2)
    interactions = make_interactions(users, recipes, candidates_per_user=30, seed=3)

    encoder = FeatureEncoder.build()
    item_features = torch.tensor(encoder.encode_items(recipes))
    user_features = torch.tensor(encoder.encode_users(users))
    item_id_to_row = {rid: i for i, rid in enumerate(recipes.recipe_id.tolist())}
    user_id_to_row = {uid: i for i, uid in enumerate(users.user_id.tolist())}

    train_df = split_positive_interactions(interactions)
    train_df = train_df[train_df.split == "train"]
    train_ds = PairDataset(train_df, user_id_to_row, item_id_to_row)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)

    model = TwoTowerModel(user_dim=encoder.user_dim, item_dim=encoder.item_dim, embed_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(25):
        for user_rows, item_rows in loader:
            loss = model.in_batch_softmax_loss(user_features[user_rows], item_features[item_rows], temperature=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    relevant = true_relevant_sets(users, recipes, k=10)
    rankings = two_tower_rankings(model, user_features, item_features, users, recipes, k=10)
    pop_top10 = popularity_ranking(recipes, k=10)

    tt_recall = np.mean([recall_at_k(rankings[uid], rel, 10) for uid, rel in relevant.items()])
    pop_recall = np.mean([recall_at_k(pop_top10, rel, 10) for rel in relevant.values()])

    assert tt_recall > pop_recall * 1.5, (
        f"two-tower recall ({tt_recall:.3f}) should clear a non-personalized popularity "
        f"baseline ({pop_recall:.3f}) by a healthy margin"
    )
