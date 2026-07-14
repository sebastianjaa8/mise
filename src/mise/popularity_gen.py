"""Generates the time-varying data the feature store demo needs.

Everything else in this project treats `pop_bias` as a static column. Real
popularity isn't static — it drifts week to week — and that drift is exactly
what makes the offline/online split and point-in-time correctness in
feature_store.py worth having. This script fabricates 8 weeks of weekly
recipe popularity snapshots and timestamps the existing interaction log
across that window, so a historical feature lookup for a week-1 interaction
can be point-in-time-correct instead of accidentally using week-8's numbers.
"""
import numpy as np
import pandas as pd

from mise.config import RANDOM_SEED

N_WEEKS = 8
WEEK_START = pd.Timestamp("2026-01-05")  # arbitrary Monday; only relative spacing matters


def make_recipe_popularity_history(recipes_df: pd.DataFrame, seed=RANDOM_SEED + 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for recipe in recipes_df.itertuples(index=False):
        trend = rng.normal(0, 0.02)  # per-week drift, some recipes trend up, some down
        value = recipe.pop_bias
        for week in range(N_WEEKS):
            value = float(np.clip(value + trend + rng.normal(0, 0.015), 0.0, 1.0))
            rows.append(dict(
                recipe_id=recipe.recipe_id,
                event_timestamp=WEEK_START + pd.Timedelta(weeks=week),
                pop_bias=round(value, 4),
                weekly_interactions=int(rng.poisson(5 + value * 40)),
            ))
    return pd.DataFrame(rows)


def make_user_profile_snapshot(users_df: pd.DataFrame) -> pd.DataFrame:
    """Static for now (one snapshot) — user profile fields don't drift in
    this dataset yet, but living in the feature store means they *can*
    without touching the model-facing FeatureEncoder."""
    df = users_df.copy()
    df["event_timestamp"] = WEEK_START
    return df[["user_id", "event_timestamp", "protein_target", "max_prep_min", "equipment", "cuisine_affinity"]]


def timestamp_interactions(interactions_df: pd.DataFrame, seed=RANDOM_SEED + 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    offsets_days = rng.integers(0, N_WEEKS * 7, size=len(interactions_df))
    df = interactions_df.copy()
    df["event_timestamp"] = WEEK_START + pd.to_timedelta(offsets_days, unit="D")
    return df


def generate_all(out_dir="feature_repo/data"):
    recipes = pd.read_csv("data/recipes.csv")
    users = pd.read_csv("data/users.csv")
    interactions = pd.read_csv("data/interactions.csv")

    popularity_history = make_recipe_popularity_history(recipes)
    user_profile = make_user_profile_snapshot(users)
    timestamped_interactions = timestamp_interactions(interactions)

    popularity_history.to_parquet(f"{out_dir}/recipe_popularity.parquet", index=False)
    user_profile.to_parquet(f"{out_dir}/user_profile.parquet", index=False)
    timestamped_interactions.to_parquet(f"{out_dir}/interactions_timestamped.parquet", index=False)

    print(f"recipe_popularity: {len(popularity_history)} rows ({N_WEEKS} weeks x {len(recipes)} recipes)")
    print(f"user_profile: {len(user_profile)} rows")
    print(f"interactions_timestamped: {len(timestamped_interactions)} rows, "
          f"spanning {timestamped_interactions.event_timestamp.min()} to {timestamped_interactions.event_timestamp.max()}")


if __name__ == "__main__":
    generate_all()
