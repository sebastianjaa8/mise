"""End-to-end smoke test for the ranking stage: retrieval pulls a candidate
pool, the LightGBM ranker reorders it, and that reorder should recover more
of the true top matches than trusting retrieval's raw similarity order alone.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mise.data_gen import make_interactions, make_recipes, make_users
from mise.dataset import FeatureEncoder, PairDataset
from mise.evaluate import true_relevant_sets, two_tower_candidates_with_scores
from mise.metrics import recall_at_k
from mise.model import TwoTowerModel
from mise.rank_features import FEATURE_COLUMNS, build_features
from mise.rerank import Reranker
from mise.train import split_positive_interactions


def _train_two_tower(users, recipes, interactions, encoder):
    item_features = torch.tensor(encoder.encode_items(recipes))
    user_features = torch.tensor(encoder.encode_users(users))
    item_id_to_row = {rid: i for i, rid in enumerate(recipes.recipe_id.tolist())}
    user_id_to_row = {uid: i for i, uid in enumerate(users.user_id.tolist())}

    train_df = split_positive_interactions(interactions)
    train_df = train_df[train_df.split == "train"]
    loader = DataLoader(PairDataset(train_df, user_id_to_row, item_id_to_row), batch_size=64, shuffle=True, drop_last=True)

    model = TwoTowerModel(user_dim=encoder.user_dim, item_dim=encoder.item_dim, embed_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(25):
        for user_rows, item_rows in loader:
            loss = model.in_batch_softmax_loss(user_features[user_rows], item_features[item_rows], temperature=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, user_features, item_features


def test_ranker_beats_retrieval_only(tmp_path):
    torch.manual_seed(0)

    recipes = make_recipes(n=150, seed=1)
    users = make_users(n=80, seed=2)
    interactions = make_interactions(users, recipes, candidates_per_user=30, seed=3)
    encoder = FeatureEncoder.build()

    model, user_features, item_features = _train_two_tower(users, recipes, interactions, encoder)
    candidates, scores = two_tower_candidates_with_scores(model, user_features, item_features, users, recipes, k=30)

    features = build_features(interactions[["user_id", "recipe_id"]], users, recipes, scores)
    features["label"] = interactions.weight.to_numpy()
    features = features.sort_values("user_id").reset_index(drop=True)
    group_sizes = features.groupby("user_id", sort=True).size().to_numpy()

    ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=50,
                             learning_rate=0.1, num_leaves=7, min_child_samples=5, verbosity=-1)
    ranker.fit(features[FEATURE_COLUMNS], features.label, group=group_sizes)
    model_path = tmp_path / "ranker.txt"
    ranker.booster_.save_model(str(model_path))

    reranker = Reranker(str(model_path))
    relevant = true_relevant_sets(users, recipes, k=10)

    retrieval_recall, ranker_recall = [], []
    for uid, rel in relevant.items():
        retrieval_top10 = candidates[uid][:10]
        reranked_top10 = reranker.rerank(uid, candidates[uid], scores, users, recipes, k=10)
        retrieval_recall.append(recall_at_k(retrieval_top10, rel, 10))
        ranker_recall.append(recall_at_k(reranked_top10, rel, 10))

    assert np.mean(ranker_recall) >= np.mean(retrieval_recall), (
        f"ranker recall ({np.mean(ranker_recall):.3f}) should be at least as good as "
        f"retrieval-order-only ({np.mean(retrieval_recall):.3f})"
    )
