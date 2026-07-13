"""Shared vocab + hyperparameters for the recipe retrieval pipeline."""

CUISINES = [
    "italian", "mexican", "indian", "mediterranean", "east_asian",
    "american", "middle_eastern", "cajun", "breakfast", "high_protein_snack",
]

EQUIPMENT = [
    "multi_cooker",   # Aroma-style rice/multi cooker
    "blender",        # Ninja
    "portable_stove",  # single burner
    "instant_pot",
    "air_fryer",
    "oven",
    "stovetop",
    "slow_cooker",
    "no_cook",
]

DIET_TAGS = [
    "high_protein", "vegetarian", "low_carb", "gluten_free", "dairy_free", "budget",
]

# Personas used to generate synthetic but structurally realistic interaction
# data. "small_kitchen_high_protein" mirrors the author's own daily-cooking
# constraints (small-footprint equipment, protein-forward, one-pot bias) so
# the dataset encodes a real preference signal instead of uniform noise.
PERSONAS = {
    "small_kitchen_high_protein": dict(
        equipment={"multi_cooker", "blender", "portable_stove", "instant_pot", "air_fryer"},
        protein_target=40,
        max_prep_min=45,
        cuisine_affinity={"mexican", "high_protein_snack", "mediterranean", "american"},
        weight=0.30,
    ),
    "home_cook_full_kitchen": dict(
        equipment={"oven", "stovetop", "slow_cooker", "blender", "air_fryer"},
        protein_target=25,
        max_prep_min=75,
        cuisine_affinity={"italian", "american", "cajun", "east_asian"},
        weight=0.20,
    ),
    "vegetarian_explorer": dict(
        equipment={"stovetop", "oven", "no_cook", "blender"},
        protein_target=15,
        max_prep_min=60,
        cuisine_affinity={"indian", "mediterranean", "east_asian"},
        weight=0.15,
    ),
    "quick_breakfast_snacker": dict(
        equipment={"blender", "no_cook", "air_fryer", "portable_stove"},
        protein_target=20,
        max_prep_min=20,
        cuisine_affinity={"breakfast", "high_protein_snack"},
        weight=0.15,
    ),
    "weekend_chef": dict(
        equipment={"oven", "stovetop", "slow_cooker", "instant_pot"},
        protein_target=30,
        max_prep_min=150,
        cuisine_affinity={"italian", "middle_eastern", "cajun", "indian"},
        weight=0.20,
    ),
}

EMBED_DIM = 32
N_RECIPES = 600
N_USERS = 400
N_EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
# Softmax temperature for in-batch-negative training. Too low (<0.1) and the
# loss collapses all item embeddings toward one direction before the towers
# ever separate classes — 0.5 is what stayed stable in a temp/lr sweep here.
TEMPERATURE = 0.5
TOP_K = 10
RANDOM_SEED = 13
