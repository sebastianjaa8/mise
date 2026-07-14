"""Offline retrieval evaluation.

Ground truth here is the simulator's noise-free preference score
(`deterministic_match_score`), not the sparse logged interactions — a real
user's 2-4 logged clicks aren't enough signal to tell a good retrieval model
from a mediocre one apart offline. Using the known-good "would this user
actually like this recipe" function as ground truth is a standard technique
for evaluating a retrieval system before real production feedback exists,
and it's what makes the retrieval-quality claim in the README defensible
instead of hand-wavy.
"""
import numpy as np
import pandas as pd
import torch

from mise.config import TOP_K
from mise.data_gen import deterministic_match_score
from mise.dataset import FeatureEncoder
from mise.metrics import ndcg_at_k, recall_at_k
from mise.model import TwoTowerModel
from mise.rerank import Reranker


def true_relevant_sets(users_df: pd.DataFrame, recipes_df: pd.DataFrame, k: int = 15) -> dict:
    """Per user: the top-k recipes by noise-free preference score."""
    relevant = {}
    for user_row in users_df.itertuples(index=False):
        scores = [deterministic_match_score(user_row, r) for r in recipes_df.itertuples(index=False)]
        top_idx = np.argsort(scores)[::-1][:k]
        relevant[user_row.user_id] = set(recipes_df.recipe_id.iloc[top_idx].tolist())
    return relevant


def two_tower_rankings(model, user_features, item_features, users_df, recipes_df, k):
    row_to_recipe_id = {i: rid for i, rid in enumerate(recipes_df.recipe_id.tolist())}
    model.eval()
    rankings = {}
    with torch.no_grad():
        item_emb = model.item_tower(item_features)
        for i, uid in enumerate(users_df.user_id.tolist()):
            user_emb = model.user_tower(user_features[i:i + 1])
            sims = (user_emb @ item_emb.T).squeeze(0)
            top_rows = torch.topk(sims, k).indices.tolist()
            rankings[uid] = [row_to_recipe_id[r] for r in top_rows]
    return rankings


def two_tower_candidates_with_scores(model, user_features, item_features, users_df, recipes_df, k):
    """Same as two_tower_rankings but also returns the raw similarity scores,
    needed as a ranker feature and for the retrieval-only baseline."""
    row_to_recipe_id = {i: rid for i, rid in enumerate(recipes_df.recipe_id.tolist())}
    model.eval()
    candidates, scores_by_user = {}, {}
    with torch.no_grad():
        item_emb = model.item_tower(item_features)
        for i, uid in enumerate(users_df.user_id.tolist()):
            user_emb = model.user_tower(user_features[i:i + 1])
            sims = (user_emb @ item_emb.T).squeeze(0)
            top_scores, top_rows = torch.topk(sims, k)
            recipe_ids = [row_to_recipe_id[r] for r in top_rows.tolist()]
            candidates[uid] = recipe_ids
            for rid, score in zip(recipe_ids, top_scores.tolist()):
                scores_by_user[(uid, rid)] = score
    return candidates, scores_by_user


def popularity_ranking(recipes_df: pd.DataFrame, k: int) -> list:
    return recipes_df.sort_values("pop_bias", ascending=False).recipe_id.head(k).tolist()


def random_ranking(recipes_df: pd.DataFrame, k: int, rng: np.random.Generator) -> list:
    return rng.choice(recipes_df.recipe_id.to_numpy(), size=k, replace=False).tolist()


def run(k=TOP_K, candidate_pool_size=50):
    import os

    recipes = pd.read_csv("data/recipes.csv")
    users = pd.read_csv("data/users.csv")

    encoder = FeatureEncoder.build()
    item_features = torch.tensor(encoder.encode_items(recipes))
    user_features = torch.tensor(encoder.encode_users(users))

    model = TwoTowerModel(user_dim=encoder.user_dim, item_dim=encoder.item_dim)
    model.load_state_dict(torch.load("artifacts/model.pt"))

    relevant = true_relevant_sets(users, recipes, k=15)
    tt_rankings = two_tower_rankings(model, user_features, item_features, users, recipes, k)
    pop_top_k = popularity_ranking(recipes, k)
    rng = np.random.default_rng(7)

    tt_recall, pop_recall, rand_recall = [], [], []
    tt_ndcg = []
    for uid, rel in relevant.items():
        tt_recall.append(recall_at_k(tt_rankings[uid], rel, k))
        tt_ndcg.append(ndcg_at_k(tt_rankings[uid], rel, k))
        pop_recall.append(recall_at_k(pop_top_k, rel, k))
        rand_recall.append(recall_at_k(random_ranking(recipes, k, rng), rel, k))

    print(f"two-tower           recall@{k}={np.mean(tt_recall):.3f}  ndcg@{k}={np.mean(tt_ndcg):.3f}")
    print(f"popularity          recall@{k}={np.mean(pop_recall):.3f}")
    print(f"random              recall@{k}={np.mean(rand_recall):.3f}")
    lift = np.mean(tt_recall) / np.mean(pop_recall) if np.mean(pop_recall) > 0 else float("inf")
    print(f"two-tower vs popularity lift: {lift:.2f}x")

    if os.path.exists("artifacts/ranker.txt"):
        _evaluate_retrieval_plus_rerank(model, user_features, item_features, users, recipes,
                                         relevant, k, candidate_pool_size)


def _evaluate_retrieval_plus_rerank(model, user_features, item_features, users_df, recipes_df,
                                     relevant, k, candidate_pool_size):
    candidates, scores = two_tower_candidates_with_scores(
        model, user_features, item_features, users_df, recipes_df, candidate_pool_size
    )
    reranker = Reranker()

    rerank_recall, rerank_ndcg = [], []
    for uid, rel in relevant.items():
        reranked = reranker.rerank(uid, candidates[uid], scores, users_df, recipes_df, k)
        rerank_recall.append(recall_at_k(reranked, rel, k))
        rerank_ndcg.append(ndcg_at_k(reranked, rel, k))

    print(f"two-tower + ranker  recall@{k}={np.mean(rerank_recall):.3f}  ndcg@{k}={np.mean(rerank_ndcg):.3f}  "
          f"(retrieval pulls top-{candidate_pool_size}, ranker cuts to top-{k})")


if __name__ == "__main__":
    run()
