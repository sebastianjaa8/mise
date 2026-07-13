"""Synthetic recipe + interaction data generator.

There's no ready-made "small kitchen, protein-forward, one appliance at a
time" recipe/interaction dataset to pull off the shelf, so this builds one
with a real preference signal baked in (equipment fit, protein target, prep
time budget, cuisine affinity) instead of uniform random noise. That's what
makes the two-tower retrieval problem non-trivial: a model that ignores
equipment/protein features should score measurably worse than one that uses
them, which is the whole point of the exercise.
"""
import random

import numpy as np
import pandas as pd

from mise.config import CUISINES, DIET_TAGS, EQUIPMENT, N_RECIPES, N_USERS, PERSONAS, RANDOM_SEED

PROTEIN_SOURCES = {
    "italian": ["chicken thigh", "ground turkey", "white bean", "shrimp"],
    "mexican": ["chicken breast", "black bean", "ground beef", "shrimp"],
    "indian": ["chickpea", "paneer", "lentil", "chicken thigh"],
    "mediterranean": ["chicken breast", "chickpea", "salmon", "lentil"],
    "east_asian": ["tofu", "chicken thigh", "ground pork", "shrimp"],
    "american": ["ground turkey", "chicken breast", "egg", "steak tip"],
    "middle_eastern": ["chickpea", "lamb", "chicken thigh", "lentil"],
    "cajun": ["shrimp", "chicken thigh", "andouille", "red bean"],
    "breakfast": ["egg", "greek yogurt", "cottage cheese", "turkey sausage"],
    "high_protein_snack": ["greek yogurt", "cottage cheese", "whey", "egg white"],
}

DISH_WORDS = [
    "bowl", "skillet", "stir-fry", "bake", "wrap", "soup", "chili", "curry",
    "salad", "scramble", "one-pot", "sheet-pan",
]

EQUIPMENT_LABEL = {
    "multi_cooker": "Multi-Cooker",
    "blender": "Blender",
    "portable_stove": "One-Burner",
    "instant_pot": "Instant Pot",
    "air_fryer": "Air Fryer",
    "oven": "Sheet-Pan",
    "stovetop": "Stovetop",
    "slow_cooker": "Slow-Cooker",
    "no_cook": "No-Cook",
}

BASE_PREP = {
    "no_cook": (5, 15),
    "blender": (5, 15),
    "portable_stove": (15, 35),
    "air_fryer": (15, 30),
    "instant_pot": (20, 45),
    "multi_cooker": (20, 50),
    "stovetop": (20, 45),
    "oven": (30, 65),
    "slow_cooker": (90, 240),
}


def _rng(seed):
    return random.Random(seed), np.random.default_rng(seed)


def make_recipes(n=N_RECIPES, seed=RANDOM_SEED):
    py_rng, np_rng = _rng(seed)
    rows = []
    for i in range(n):
        cuisine = py_rng.choice(CUISINES)
        primary_equipment = py_rng.choice(list(BASE_PREP.keys()))
        lo, hi = BASE_PREP[primary_equipment]
        prep_time = int(np_rng.integers(lo, hi + 1))

        # ~35% of the catalog is deliberately biased toward the
        # small-kitchen / high-protein persona: one appliance, protein-heavy,
        # quick. The rest of the catalog is a normal spread so the model has
        # to actually learn the fit rather than memorize a global prior.
        skew_small_kitchen = py_rng.random() < 0.35
        if skew_small_kitchen:
            primary_equipment = py_rng.choice(["multi_cooker", "blender", "portable_stove", "instant_pot", "air_fryer"])
            lo, hi = BASE_PREP[primary_equipment]
            prep_time = int(np_rng.integers(lo, min(hi, 45) + 1))
            protein_g = int(np_rng.integers(30, 55))
        else:
            protein_g = int(np_rng.integers(8, 40))

        equipment = {primary_equipment}
        if py_rng.random() < 0.25:
            equipment.add(py_rng.choice(EQUIPMENT))

        diet_tags = set()
        if protein_g >= 30:
            diet_tags.add("high_protein")
        if py_rng.random() < 0.2:
            diet_tags.add("vegetarian")
        if py_rng.random() < 0.15:
            diet_tags.add("low_carb")
        if py_rng.random() < 0.1:
            diet_tags.add("gluten_free")
        if py_rng.random() < 0.15:
            diet_tags.add("budget")

        protein_word = py_rng.choice(PROTEIN_SOURCES[cuisine])
        dish = py_rng.choice(DISH_WORDS)
        title = f"{EQUIPMENT_LABEL[primary_equipment]} {protein_word.title()} {dish.title()}"

        rows.append(dict(
            recipe_id=i,
            title=title,
            cuisine=cuisine,
            protein_g=protein_g,
            prep_time_min=prep_time,
            equipment="|".join(sorted(equipment)),
            diet_tags="|".join(sorted(diet_tags)),
            pop_bias=round(float(np_rng.beta(2, 5)), 4),  # long-tail baseline popularity
        ))
    return pd.DataFrame(rows)


def make_users(n=N_USERS, seed=RANDOM_SEED + 1):
    py_rng, np_rng = _rng(seed)
    persona_names = list(PERSONAS.keys())
    weights = [PERSONAS[p]["weight"] for p in persona_names]

    rows = []
    for uid in range(n):
        persona_name = py_rng.choices(persona_names, weights=weights, k=1)[0]
        persona = PERSONAS[persona_name]

        equipment = set(persona["equipment"])
        if py_rng.random() < 0.15 and len(equipment) > 1:
            equipment.discard(py_rng.choice(list(equipment)))
        if py_rng.random() < 0.15:
            equipment.add(py_rng.choice(EQUIPMENT))

        protein_target = max(10, int(persona["protein_target"] + np_rng.normal(0, 6)))
        max_prep_min = max(10, int(persona["max_prep_min"] + np_rng.normal(0, 10)))

        cuisine_affinity = set(persona["cuisine_affinity"])
        if py_rng.random() < 0.3:
            cuisine_affinity.add(py_rng.choice(CUISINES))

        rows.append(dict(
            user_id=uid,
            persona=persona_name,
            protein_target=protein_target,
            max_prep_min=max_prep_min,
            equipment="|".join(sorted(equipment)),
            cuisine_affinity="|".join(sorted(cuisine_affinity)),
        ))
    return pd.DataFrame(rows)


def deterministic_match_score(user_row, recipe_row) -> float:
    """Noise-free preference score. This is the "ground truth" simulator
    function: `_match_score` below adds noise on top of this to produce the
    logged interactions a real system would actually observe. Evaluation
    uses this directly (see evaluate.py) because logged implicit feedback is
    sparse and noisy — 2-4 held-out clicks per user isn't enough to reliably
    tell a good retrieval model from a mediocre one, so offline eval here
    checks retrieval against the full simulator-known preference set instead
    of just the handful of interactions that happened to get logged.
    """
    user_equipment = set(user_row.equipment.split("|")) if user_row.equipment else set()
    recipe_equipment = set(recipe_row.equipment.split("|")) if recipe_row.equipment else set()
    equipment_fit = 1.0 if recipe_equipment & user_equipment else 0.0

    protein_fit = 1.0 - min(1.0, abs(recipe_row.protein_g - user_row.protein_target) / 40.0)
    prep_fit = 1.0 if recipe_row.prep_time_min <= user_row.max_prep_min else max(
        0.0, 1.0 - (recipe_row.prep_time_min - user_row.max_prep_min) / 60.0
    )
    cuisine_affinity = set(user_row.cuisine_affinity.split("|")) if user_row.cuisine_affinity else set()
    cuisine_fit = 1.0 if recipe_row.cuisine in cuisine_affinity else 0.2

    score = (
        0.35 * equipment_fit
        + 0.30 * protein_fit
        + 0.15 * prep_fit
        + 0.20 * cuisine_fit
    )
    return float(np.clip(score, 0, 1))


def _match_score(user_row, recipe_row, np_rng):
    score = deterministic_match_score(user_row, recipe_row)
    score += np_rng.normal(0, 0.08)  # real users aren't perfectly rational
    return float(np.clip(score, 0, 1))


def make_interactions(users_df, recipes_df, candidates_per_user=40, seed=RANDOM_SEED + 2):
    py_rng, np_rng = _rng(seed)
    rows = []
    recipe_ids = recipes_df.recipe_id.tolist()

    for user_row in users_df.itertuples(index=False):
        candidate_ids = py_rng.sample(recipe_ids, k=min(candidates_per_user, len(recipe_ids)))
        for rid in candidate_ids:
            recipe_row = recipes_df.loc[recipes_df.recipe_id == rid].iloc[0]
            score = _match_score(user_row, recipe_row, np_rng)

            # 5% pure exploration: users occasionally cook something outside
            # their usual pattern. Keeps the label distribution from being a
            # deterministic function of the features.
            if py_rng.random() < 0.05:
                score = float(np_rng.uniform(0, 1))

            if score > 0.72:
                label, weight = "cooked", 3
            elif score > 0.58:
                label, weight = "saved", 1
            else:
                label, weight = "skipped", 0

            rows.append(dict(
                user_id=user_row.user_id,
                recipe_id=rid,
                label=label,
                weight=weight,
                is_positive=int(weight > 0),
            ))
    return pd.DataFrame(rows)


def generate_all(out_dir="data"):
    recipes = make_recipes()
    users = make_users()
    interactions = make_interactions(users, recipes)

    recipes.to_csv(f"{out_dir}/recipes.csv", index=False)
    users.to_csv(f"{out_dir}/users.csv", index=False)
    interactions.to_csv(f"{out_dir}/interactions.csv", index=False)

    pos_rate = interactions.is_positive.mean()
    print(f"recipes={len(recipes)} users={len(users)} interactions={len(interactions)} "
          f"positive_rate={pos_rate:.3f}")
    return recipes, users, interactions


if __name__ == "__main__":
    generate_all()
