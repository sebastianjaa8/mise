"""Hand-crafted (user, recipe) pair features for the ranking stage.

The retrieval stage only ever sees a dense embedding similarity — it's
cheap to run over the whole catalog but throws away interpretable signal
(exact protein gap, exact prep-time slack) that's easy to compute once
you're down to a shortlist of ~50 candidates. The ranker gets both: the
retrieval model's own similarity score AND these explicit features, which
is the standard "cheap+broad retrieval, expensive+narrow ranking" split.
"""
import numpy as np
import pandas as pd

from mise.config import DIET_TAGS


def _set(value: str) -> set:
    return set(value.split("|")) if isinstance(value, str) and value else set()


def build_features(pairs_df: pd.DataFrame, users_df: pd.DataFrame, recipes_df: pd.DataFrame,
                    two_tower_scores: dict) -> pd.DataFrame:
    """pairs_df needs columns: user_id, recipe_id.
    two_tower_scores: dict[(user_id, recipe_id)] -> float cosine similarity.
    """
    users_idx = users_df.set_index("user_id")
    recipes_idx = recipes_df.set_index("recipe_id")

    rows = []
    for user_id, recipe_id in zip(pairs_df.user_id, pairs_df.recipe_id):
        u = users_idx.loc[user_id]
        r = recipes_idx.loc[recipe_id]

        user_equipment = _set(u.equipment)
        recipe_equipment = _set(r.equipment)
        user_cuisine_aff = _set(u.cuisine_affinity)
        recipe_diet = _set(r.diet_tags)

        rows.append(dict(
            user_id=user_id,
            recipe_id=recipe_id,
            two_tower_score=two_tower_scores.get((user_id, recipe_id), 0.0),
            equipment_overlap=len(user_equipment & recipe_equipment),
            protein_diff=abs(r.protein_g - u.protein_target),
            prep_slack=u.max_prep_min - r.prep_time_min,
            cuisine_match=int(r.cuisine in user_cuisine_aff),
            recipe_diet_tag_count=len(recipe_diet),  # catalog richness signal; users have no explicit diet-tag prefs field yet
            pop_bias=r.pop_bias,
            recipe_protein_g=r.protein_g,
            recipe_prep_time_min=r.prep_time_min,
        ))
    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "two_tower_score", "equipment_overlap", "protein_diff", "prep_slack",
    "cuisine_match", "recipe_diet_tag_count", "pop_bias", "recipe_protein_g", "recipe_prep_time_min",
]
