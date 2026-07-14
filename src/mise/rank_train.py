"""Train the LightGBM ranking stage on top of retrieval candidates.

Two-tower retrieval answers "which ~50 recipes out of the whole catalog are
even worth considering" (cheap, broad). This stage answers "given those ~50,
what's the best order" using richer explicit features plus the retrieval
model's own similarity score — the standard candidate-generation + ranking
split in production recsys.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from mise.config import RANDOM_SEED
from mise.dataset import FeatureEncoder
from mise.metrics import ndcg_at_k, recall_at_k
from mise.model import TwoTowerModel
from mise.rank_features import FEATURE_COLUMNS, build_features


def _split_users(user_ids: list, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(user_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_val, n_test = int(n * 0.15), int(n * 0.15)
    return dict(
        train=set(ids[: n - n_val - n_test]),
        val=set(ids[n - n_val - n_test: n - n_test]),
        test=set(ids[n - n_test:]),
    )


def _compute_two_tower_scores(users_df, recipes_df, pairs_df) -> dict:
    encoder = FeatureEncoder.build()
    model = TwoTowerModel(user_dim=encoder.user_dim, item_dim=encoder.item_dim)
    model.load_state_dict(torch.load("artifacts/model.pt"))
    model.eval()

    item_embeddings = np.load("artifacts/item_embeddings.npy")
    item_ids = np.load("artifacts/item_ids.npy")
    item_row = {int(rid): i for i, rid in enumerate(item_ids)}

    user_features = torch.tensor(encoder.encode_users(users_df))
    user_row = {uid: i for i, uid in enumerate(users_df.user_id.tolist())}

    with torch.no_grad():
        user_embeddings = model.user_tower(user_features).numpy()

    scores = {}
    for user_id, recipe_id in zip(pairs_df.user_id, pairs_df.recipe_id):
        u_emb = user_embeddings[user_row[user_id]]
        i_emb = item_embeddings[item_row[recipe_id]]
        scores[(user_id, recipe_id)] = float(np.dot(u_emb, i_emb))
    return scores


def _grouped(df: pd.DataFrame):
    """Sort by user_id and return (df, group_sizes) as LightGBM's ranking
    API requires contiguous per-query groups."""
    df = df.sort_values("user_id").reset_index(drop=True)
    group_sizes = df.groupby("user_id", sort=True).size().to_numpy()
    return df, group_sizes


def train():
    recipes = pd.read_csv("data/recipes.csv")
    users = pd.read_csv("data/users.csv")
    interactions = pd.read_csv("data/interactions.csv")

    splits = _split_users(users.user_id.tolist())
    two_tower_scores = _compute_two_tower_scores(users, recipes, interactions)

    features = build_features(interactions[["user_id", "recipe_id"]], users, recipes, two_tower_scores)
    features["label"] = interactions.weight.to_numpy()

    train_df, train_groups = _grouped(features[features.user_id.isin(splits["train"])])
    val_df, val_groups = _grouped(features[features.user_id.isin(splits["val"])])
    test_df, test_groups = _grouped(features[features.user_id.isin(splits["test"])])

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        random_state=RANDOM_SEED,
        verbosity=-1,
    )
    ranker.fit(
        train_df[FEATURE_COLUMNS], train_df.label,
        group=train_groups,
        eval_set=[(val_df[FEATURE_COLUMNS], val_df.label)],
        eval_group=[val_groups],
        eval_at=[10],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    test_df = test_df.copy()
    test_df["ranker_score"] = ranker.predict(test_df[FEATURE_COLUMNS])
    test_ndcg = np.mean([
        ndcg_at_k(group.sort_values("ranker_score", ascending=False).recipe_id.tolist(),
                  set(group.loc[group.label > 0, "recipe_id"]), 10)
        for _, group in test_df.groupby("user_id") if (group.label > 0).any()
    ])
    print(f"ranker held-out-user ndcg@10 (vs its own logged candidate labels)={test_ndcg:.3f}")
    print("see `python -m mise.evaluate` for the retrieval-vs-retrieval+ranker end-to-end comparison")

    ranker.booster_.save_model("artifacts/ranker.txt")
    print("saved artifacts/ranker.txt")
    return ranker


if __name__ == "__main__":
    train()
