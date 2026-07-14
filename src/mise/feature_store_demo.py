"""Demonstrates the two things a feature store is actually for here:

1. Point-in-time-correct historical joins for training data (an interaction
   from week 1 should see week 1's recipe popularity, not week 8's — using
   "whatever the popularity is right now" for all of training history is a
   classic, subtle training/serving-skew bug).
2. A materialized online store for low-latency feature lookups at request
   time, fed from the same source the offline join used.

Run `python -m mise.popularity_gen` and `feast apply` (from feature_repo/)
before this.
"""
import pandas as pd
from feast import FeatureStore


def _naive_current_snapshot_join(interactions: pd.DataFrame, popularity_history: pd.DataFrame) -> pd.DataFrame:
    """What you'd get if you (incorrectly) joined every historical
    interaction against today's popularity value instead of the value that
    existed at interaction time."""
    latest = popularity_history.sort_values("event_timestamp").groupby("recipe_id").tail(1)
    return interactions.merge(latest[["recipe_id", "pop_bias"]], on="recipe_id", how="left")


def run():
    store = FeatureStore(repo_path="feature_repo")

    interactions = pd.read_parquet("feature_repo/data/interactions_timestamped.parquet")
    popularity_history = pd.read_parquet("feature_repo/data/recipe_popularity.parquet")

    sample = interactions.sample(n=500, random_state=0)[["user_id", "recipe_id", "event_timestamp"]].copy()
    sample["event_timestamp"] = pd.to_datetime(sample["event_timestamp"], utc=True)

    point_in_time = store.get_historical_features(
        entity_df=sample,
        features=["recipe_popularity:pop_bias", "recipe_popularity:weekly_interactions"],
    ).to_df()

    naive = _naive_current_snapshot_join(sample, popularity_history)

    merged = point_in_time.merge(
        naive[["user_id", "recipe_id", "event_timestamp", "pop_bias"]],
        on=["user_id", "recipe_id", "event_timestamp"],
        suffixes=("_point_in_time", "_naive_latest"),
    )
    merged["diff"] = (merged.pop_bias_point_in_time - merged.pop_bias_naive_latest).abs()
    leaked = merged[merged["diff"] > 0.02]

    print(f"sampled {len(sample)} historical interactions")
    print(f"point-in-time vs naive-latest-snapshot disagree on {len(leaked)} rows "
          f"({len(leaked) / len(merged):.1%}) by more than 0.02 pop_bias")
    if len(leaked):
        print("example (naive join would have used a popularity value the recipe didn't have yet):")
        print(leaked[["recipe_id", "event_timestamp", "pop_bias_point_in_time", "pop_bias_naive_latest"]].head(3))

    print("\nmaterializing online store...")
    # First-time full materialize (not `materialize_incremental`, which tracks
    # a "last materialized" watermark per feature view — that watermark
    # defaults to apply-time, which is *after* this fabricated historical
    # data, so an incremental call here would compute an empty window).
    store.materialize(
        start_date=popularity_history.event_timestamp.min(),
        end_date=popularity_history.event_timestamp.max(),
    )

    sample_recipe_ids = interactions.recipe_id.drop_duplicates().head(3).tolist()
    online = store.get_online_features(
        features=["recipe_popularity:pop_bias", "recipe_popularity:weekly_interactions"],
        entity_rows=[{"recipe_id": rid} for rid in sample_recipe_ids],
    ).to_dict()
    print("online feature lookup (as of latest materialized snapshot):")
    for i, rid in enumerate(sample_recipe_ids):
        print(f"  recipe_id={rid}  pop_bias={online['pop_bias'][i]:.3f}  "
              f"weekly_interactions={online['weekly_interactions'][i]}")


if __name__ == "__main__":
    run()
