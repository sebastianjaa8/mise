"""Feast feature definitions for mise.

Two feature views: `recipe_popularity` (time-varying — 8 weekly snapshots
per recipe, the one that actually needs point-in-time correctness) and
`user_profile` (a single static snapshot today, but living in the feature
store from day one means it can start drifting weekly without the model
code caring where the numbers came from).
"""
from datetime import timedelta

from feast import Entity, Field, FeatureView, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

recipe = Entity(name="recipe_id", join_keys=["recipe_id"], value_type=ValueType.INT64)
user = Entity(name="user_id", join_keys=["user_id"], value_type=ValueType.INT64)

recipe_popularity_source = FileSource(
    path="data/recipe_popularity.parquet",
    timestamp_field="event_timestamp",
)

recipe_popularity_fv = FeatureView(
    name="recipe_popularity",
    entities=[recipe],
    ttl=timedelta(days=14),
    schema=[
        Field(name="pop_bias", dtype=Float32),
        Field(name="weekly_interactions", dtype=Int64),
    ],
    source=recipe_popularity_source,
    online=True,
)

user_profile_source = FileSource(
    path="data/user_profile.parquet",
    timestamp_field="event_timestamp",
)

user_profile_fv = FeatureView(
    name="user_profile",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="protein_target", dtype=Int64),
        Field(name="max_prep_min", dtype=Int64),
        Field(name="equipment", dtype=String),
        Field(name="cuisine_affinity", dtype=String),
    ],
    source=user_profile_source,
    online=True,
)
