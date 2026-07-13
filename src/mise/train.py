"""Train the two-tower retrieval model on the synthetic recipe/interaction
data, then dump the artifacts the serving layer needs: the user tower
weights (for real-time query encoding) and precomputed item embeddings (for
offline index build) — precomputing item embeddings once and only encoding
the user online is the standard two-tower serving split, since the item
catalog changes far less often than a request stream.
"""
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mise.config import BATCH_SIZE, EMBED_DIM, LEARNING_RATE, N_EPOCHS, RANDOM_SEED, TEMPERATURE, TOP_K, WEIGHT_DECAY
from mise.data_gen import deterministic_match_score
from mise.dataset import FeatureEncoder, PairDataset
from mise.model import TwoTowerModel

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def split_positive_interactions(interactions_df: pd.DataFrame, seed=RANDOM_SEED):
    """Per-user 70/15/15 split of positive interactions so every active user
    can contribute to train, val, and test (users with a single positive
    interaction fall through to train only — nothing to hold out for them)."""
    rng = np.random.default_rng(seed)
    positives = interactions_df[interactions_df.is_positive == 1].copy()
    split = np.empty(len(positives), dtype=object)

    idx_by_user = positives.groupby("user_id").indices
    pos_index = positives.index.to_numpy()
    for _, row_positions in idx_by_user.items():
        row_positions = np.array(row_positions)
        rng.shuffle(row_positions)
        n = len(row_positions)
        n_val = max(0, int(n * 0.15)) if n >= 4 else 0
        n_test = max(0, int(n * 0.15)) if n >= 4 else 0
        n_train = n - n_val - n_test
        labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        for pos, label in zip(row_positions, labels):
            split[pos] = label

    positives["split"] = split
    return positives


def _true_relevant_sets(users_df: pd.DataFrame, recipes_df: pd.DataFrame, k: int = 15) -> dict:
    """Ground-truth (noise-free simulator score) top-k per user, used only
    to monitor training progress. See evaluate.py for the full writeup of
    why this beats evaluating against sparse logged interactions."""
    relevant = {}
    for user_row in users_df.itertuples(index=False):
        scores = [deterministic_match_score(user_row, r) for r in recipes_df.itertuples(index=False)]
        top_idx = np.argsort(scores)[::-1][:k]
        relevant[user_row.user_id] = set(recipes_df.recipe_id.iloc[top_idx].tolist())
    return relevant


def train():
    recipes = pd.read_csv("data/recipes.csv")
    users = pd.read_csv("data/users.csv")
    interactions = pd.read_csv("data/interactions.csv")

    encoder = FeatureEncoder.build()
    item_features = torch.tensor(encoder.encode_items(recipes))
    user_features = torch.tensor(encoder.encode_users(users))

    item_id_to_row = {rid: i for i, rid in enumerate(recipes.recipe_id.tolist())}
    user_id_to_row = {uid: i for i, uid in enumerate(users.user_id.tolist())}

    split_positives = split_positive_interactions(interactions)
    train_df = split_positives[split_positives.split == "train"]

    train_ds = PairDataset(train_df, user_id_to_row, item_id_to_row)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = TwoTowerModel(user_dim=encoder.user_dim, item_dim=encoder.item_dim, embed_dim=EMBED_DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    row_to_recipe_id = {i: rid for i, rid in enumerate(recipes.recipe_id.tolist())}
    ground_truth_relevant = _true_relevant_sets(users, recipes, k=15)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for user_rows, item_rows in train_loader:
            user_x = user_features[user_rows]
            item_x = item_features[item_rows]
            loss = model.in_batch_softmax_loss(user_x, item_x, temperature=TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(user_rows)

        avg_loss = total_loss / len(train_ds)
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            recall = _val_recall(model, user_features, item_features, ground_truth_relevant, user_id_to_row, row_to_recipe_id, k=TOP_K)
            print(f"epoch {epoch:02d}  train_loss={avg_loss:.4f}  recall@{TOP_K} (vs sim ground truth)={recall:.3f}")
        else:
            print(f"epoch {epoch:02d}  train_loss={avg_loss:.4f}")

    _save_artifacts(model, item_features, recipes)
    return model


def _val_recall(model, user_features, item_features, val_relevant, user_id_to_row, row_to_recipe_id, k):
    from mise.metrics import recall_at_k

    if not val_relevant:
        return float("nan")

    model.eval()
    with torch.no_grad():
        all_item_emb = model.item_tower(item_features)
        scores = []
        for uid, relevant in val_relevant.items():
            row = user_id_to_row[uid]
            user_emb = model.user_tower(user_features[row:row + 1])
            sims = (user_emb @ all_item_emb.T).squeeze(0)
            topk_rows = torch.topk(sims, k).indices.tolist()
            retrieved_ids = [row_to_recipe_id[r] for r in topk_rows]
            scores.append(recall_at_k(retrieved_ids, relevant, k))
    return float(np.nanmean(scores)) if scores else float("nan")


def _save_artifacts(model, item_features, recipes_df):
    import os
    os.makedirs("artifacts", exist_ok=True)

    model.eval()
    with torch.no_grad():
        item_embeddings = model.item_tower(item_features).numpy().astype("float32")

    np.save("artifacts/item_embeddings.npy", item_embeddings)
    np.save("artifacts/item_ids.npy", recipes_df.recipe_id.to_numpy())
    torch.save(model.state_dict(), "artifacts/model.pt")
    print(f"saved artifacts/: item_embeddings{item_embeddings.shape}, model.pt")


if __name__ == "__main__":
    train()
