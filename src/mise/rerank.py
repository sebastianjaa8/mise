"""Ranking-stage inference: take retrieval's candidate shortlist and reorder
it with the LightGBM model + explicit features, instead of trusting raw
embedding similarity alone."""
import lightgbm as lgb
import pandas as pd

from mise.rank_features import FEATURE_COLUMNS, build_features


class Reranker:
    def __init__(self, model_path="artifacts/ranker.txt"):
        self.booster = lgb.Booster(model_file=model_path)

    def rerank(self, user_id: int, candidate_recipe_ids: list, two_tower_scores: dict,
               users_df, recipes_df, k: int) -> list:
        pairs = pd.DataFrame({"user_id": [user_id] * len(candidate_recipe_ids),
                              "recipe_id": candidate_recipe_ids})
        features = build_features(pairs, users_df, recipes_df, two_tower_scores)
        features["ranker_score"] = self.booster.predict(features[FEATURE_COLUMNS])
        ranked = features.sort_values("ranker_score", ascending=False)
        return ranked.recipe_id.head(k).tolist()
