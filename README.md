# mise

A recipe recommendation engine built around a real constraint: one multi-cooker,
one blender, one portable burner, and a standing goal of getting enough protein
in without a full kitchen to work with. Most recipe apps assume you have every
appliance and cook for taste first; this one starts from "what can I actually
make on this equipment, at this protein target, in the time I have" and treats
that as the retrieval problem.

It's also a working excuse to build the infra pattern behind large-scale
recommenders (two-tower retrieval, ANN search, a feature store, a ranking
stage, and real serving/monitoring) end to end on a scale I can actually run
and reason about on a laptop, instead of importing a black-box library.

## What's implemented right now

- **Two-tower retrieval model** (PyTorch): a user tower (equipment owned,
  protein target, prep-time budget, cuisine affinity) and an item tower
  (recipe cuisine, protein, prep time, equipment required, diet tags),
  trained with in-batch sampled softmax so every other recipe in a batch is a
  free negative for the model to push away from.
- **FAISS retrieval index**: item embeddings are precomputed offline (the
  catalog changes rarely); a user embedding is computed at request time and
  matched against the index with inner-product search.
- **LightGBM ranking stage**: retrieval pulls a top-50 candidate pool, the
  ranker reorders it using hand-crafted features (equipment overlap, protein
  gap, prep-time slack, cuisine match, catalog popularity) *plus* the
  retrieval model's own similarity score, trained with `lambdarank` on logged
  implicit feedback grouped per user.
- **FastAPI serving layer**: `/recommend?user_id=...` for known users
  (goes through the full retrieval→rerank pipeline),
  `/recommend?equipment=...&protein_target=...` for a brand-new user with no
  history yet — retrieval-only, since the ranker's features need a
  users.csv row that a not-yet-onboarded profile doesn't have.
- **Offline eval harness** that checks retrieval quality against a
  simulator ground truth, not just the sparse logged interactions (see
  "Data & evaluation methodology" below for why).

## Not built yet (tracked, not hidden)

- [ ] Feast feature store (offline/online split, point-in-time correctness)
- [ ] Candidate-gen / ranking split into separate services + async feature
      recompute queue
- [ ] AWS deploy (ECS Fargate) + load test (p99 latency under concurrent load)
- [ ] Prometheus/Grafana serving metrics

## Architecture

```
                         ┌─────────────────────┐
                         │   recipe catalog      │
                         │  (cuisine, protein,   │
                         │  equipment, prep time)│
                         └──────────┬────────────┘
                                    │ offline, on catalog change
                                    ▼
                         ┌─────────────────────┐
                         │     item tower        │──► item embeddings (32-d)
                         └─────────────────────┘          │
                                                           ▼
                                                 ┌───────────────────┐
  user profile ──► user tower ──► user embedding │  FAISS IndexFlatIP │
  (equipment,          (online,                  └───────────────────┘
   protein target,      per request)                       │
   prep budget,                                             ▼
   cuisine affinity)                                 top-50 candidates
                                                             │
                                                             ▼
                                              ┌────────────────────────────┐
                                              │  LightGBM ranker (lambdarank)│
                                              │  two-tower score + equipment │
                                              │  overlap + protein/prep fit  │
                                              │  + cuisine match + popularity│
                                              └────────────────────────────┘
                                                             │
                                                             ▼
                                                       top-10 recipe ids
                                                             │
                                                             ▼
                                                     FastAPI /recommend
```

## Data & evaluation methodology

There's no existing "small kitchen, protein-forward" recipe interaction
dataset to pull off the shelf, so `data_gen.py` builds one: a recipe catalog
and a set of user personas (mine included — small-kitchen, high-protein,
quick-prep) with a scored preference function (equipment fit + protein fit +
prep-time fit + cuisine affinity + noise) that produces the logged
`view / save / cook` interactions the model trains on.

Because any one user only logs a handful of interactions, evaluating
retrieval quality against just those held-out clicks is noisy — a good model
and a mediocre one look similar with only 2-4 relevant items per user to
check against. So `evaluate.py` scores retrieval against the *noise-free*
version of the same preference function (the simulator's "true" top-15
matches for that user), which is a standard trick for evaluating a retrieval
system before it has a real production feedback loop. Current numbers:

| Ranking source              | Recall@10 | Lift vs. popularity |
|-------------------------------|-----------|----------------------|
| Two-tower retrieval + ranker  | ~0.205    | 16x                   |
| Two-tower retrieval alone     | ~0.087    | 7x                    |
| Popularity baseline            | ~0.013    | 1x                    |
| Random                         | ~0.027    | 2x                    |

(popularity baseline = same top-10 recipes for every user, ignoring their
profile entirely. The ranker's ~2.4x lift over retrieval-alone comes from
recovering true matches that landed in positions 11-50 by raw embedding
similarity but get pulled back into the top-10 once explicit features like
exact protein gap and equipment overlap are in play — the retrieval stage's
job is "don't miss anything plausible," the ranker's job is "get the order
right", and the two numbers here show each one earning its place in the
pipeline)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt

python -m mise.data_gen        # writes data/recipes.csv, users.csv, interactions.csv
python -m mise.train           # trains the two-tower model -> artifacts/model.pt + item_embeddings.npy
python -m mise.build_index     # builds artifacts/items.faiss from the item embeddings
python -m mise.rank_train      # trains the LightGBM ranker -> artifacts/ranker.txt
python -m mise.evaluate        # retrieval-only vs. retrieval+ranker vs. popularity/random baselines

uvicorn mise.api:app --reload  # http://localhost:8000/recommend?user_id=0
```

Run the smoke test:

```bash
pytest tests/
```

## Repo layout

```
src/mise/
  config.py         vocab + hyperparameters
  data_gen.py       synthetic recipe catalog + user personas + interaction log
  dataset.py        feature encoding, PyTorch Dataset
  model.py          two-tower model + in-batch softmax loss
  train.py          two-tower training loop
  build_index.py    FAISS index build from precomputed item embeddings
  rank_features.py  hand-crafted (user, recipe) pair features for the ranker
  rank_train.py     LightGBM lambdarank training loop
  rerank.py         query-time ranking-stage inference
  retrieve.py       query-time recommender (retrieval + rerank, known user + cold-start profile)
  evaluate.py       retrieval-only vs. retrieval+ranker vs. baselines
  api.py            FastAPI serving layer
tests/
  test_pipeline.py  end-to-end: generate -> train -> index -> retrieve -> assert lift over popularity
  test_ranking.py   end-to-end: retrieval + ranker -> assert ranker doesn't regress retrieval order
```
