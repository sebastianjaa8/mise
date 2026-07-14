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
- **Feast feature store** (`feature_repo/`): recipe popularity is modeled as
  a genuinely time-varying feature (8 weekly snapshots, not a static column),
  with an offline file store for point-in-time-correct training joins and a
  materialized SQLite online store for request-time lookups. See "Why a
  feature store" below — it catches a real leak, not a hypothetical one.
- **Split candidate-gen + ranking services** (`src/services/`): the
  monolithic `mise.api` still exists as the simple reference
  implementation, but `candidate_service` (retrieval) and `ranking_service`
  (reranking) also run as two independent FastAPI apps, composed by a
  `gateway` over real HTTP calls — each has its own cost profile (one FAISS
  lookup vs. a GBDT pass per candidate) and can scale independently.
- **Async catalog-refresh queue**: an admin endpoint enqueues an index
  rebuild instead of doing it inline on the request thread; a separate
  worker process claims and processes it from a durable SQLite-backed
  queue, and a hot-reload endpoint swaps the new index in without
  restarting the service.
- **Containerized + Cloud Run deploy config** (`deploy/`): one lean
  Dockerfile per service (each installs only the dependencies that service
  actually needs — the gateway image has no ML libraries in it at all) and
  a `deploy.sh` that builds via Cloud Build and deploys all three to Cloud
  Run (no local Docker required to ship this).
- **Load test script + Prometheus `/metrics` on every service** — see "Load
  test & monitoring" below for real numbers and what they show.

## Not built yet (tracked, not hidden)

- [ ] Actual live Cloud Run deployment (script is written and the local
      3-service stack is verified; blocked on `gcloud auth login`, which
      needs an interactive browser login — not something a headless
      process can do. See `deploy/deploy.sh`.)

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
| Two-tower retrieval + ranker  | ~0.22     | ~17x                  |
| Two-tower retrieval alone     | ~0.07-0.09| ~6-7x                 |
| Popularity baseline            | ~0.013    | 1x                    |
| Random                         | ~0.02-0.03| ~2x                   |

(numbers vary a little run to run — floating-point non-determinism in
multi-threaded training ops means `python -m mise.train` isn't bit-for-bit
reproducible even with a fixed seed; the relative story — ranker clearly
beats retrieval alone, retrieval alone clearly beats popularity — holds
across runs, which is what actually matters here)

(popularity baseline = same top-10 recipes for every user, ignoring their
profile entirely. The ranker's further lift over retrieval-alone comes from
recovering true matches that landed in positions 11-50 by raw embedding
similarity but get pulled back into the top-10 once explicit features like
exact protein gap and equipment overlap are in play — the retrieval stage's
job is "don't miss anything plausible," the ranker's job is "get the order
right", and the two numbers here show each one earning its place in the
pipeline)

## Why a feature store

Everywhere else in this repo, `pop_bias` is a static column. That's fine for
the retrieval/ranking demo, but it's not how popularity actually behaves —
recipes trend up and down week to week. `feature_repo/` fabricates 8 weekly
popularity snapshots per recipe and timestamps the interaction log across
that window, which makes a real mistake possible: joining *every* historical
training row against *today's* popularity value instead of the value that
recipe actually had at interaction time. That's future information leaking
into the training set — a model trained on it looks great offline and
underperforms in production, because at serving time the "future" value
obviously isn't available yet.

`feature_store_demo.py` builds both versions side by side — Feast's
point-in-time-correct historical join, and a naive "current snapshot" join —
on the same 500 sampled historical interactions:

```
point-in-time vs naive-latest-snapshot disagree on 319 rows (64.3%)
```

Almost two-thirds of sampled rows would have trained on a popularity value
that didn't exist yet. The same `feature_repo/` also materializes those
snapshots into a local SQLite online store, so a request-time lookup at
serving time is a real feature-store call, not a CSV read — the offline
training join and the online serving lookup pull from the same source
instead of two paths that can quietly drift apart.

## Service split + async catalog refresh

`src/services/candidate_service.py` and `src/services/ranking_service.py`
are the same retrieval and reranking logic as `mise.api`, but as two
independent FastAPI apps instead of one process — `src/services/gateway.py`
is the only thing a client talks to, and it composes the other two over
real HTTP:

```bash
uvicorn services.candidate_service:app --port 8001
uvicorn services.ranking_service:app   --port 8002
CANDIDATE_SERVICE_URL=http://localhost:8001 RANKING_SERVICE_URL=http://localhost:8002 \
  uvicorn services.gateway:app --port 8000

curl "http://localhost:8000/recommend?user_id=0&k=5"
```

Catalog refresh (new recipes added, embeddings need recomputing) shouldn't
block a request thread, so it goes through a queue instead of running
inline:

```bash
curl -X POST http://localhost:8001/admin/refresh-catalog   # returns a job_id immediately
python -m mise.worker --once                                # a separate process claims + processes the job
curl -X POST http://localhost:8001/admin/reload-index        # hot-swap the new index in, no restart
```

`mise.job_queue` is SQLite-backed rather than a real broker (SQS/Kafka/Redis
Streams) — same durability property (survives a process restart, safe under
polling) without a broker dependency for a single-node demo; swapping the
backend later means replacing that one file, not any caller.

## Deploying (Cloud Run)

Three lean, per-service Dockerfiles (`deploy/Dockerfile.{candidate,ranking,gateway}`)
plus matching per-service requirement lists — the gateway image, for
example, has no torch/faiss/lightgbm in it at all, since all it ever does is
proxy JSON between the other two:

```bash
deploy/deploy.sh <gcp-project-id> [region]   # builds via Cloud Build, deploys all 3 to Cloud Run
```

`gcloud run deploy` + `--source`/`--tag` builds server-side via Cloud Build,
so this doesn't need Docker installed locally. Cloud Run's Always Free tier
(2M requests/month, 360k vCPU-seconds) covers a demo workload at zero cost,
and `--min-instances 0` means it scales to zero (and $0) between uses.

## Load test & monitoring

Every service exposes Prometheus-format metrics at `/metrics`
(`prometheus-fastapi-instrumentator` — request count, latency histogram,
and status code, per route). Cloud Run also auto-exports its own
infra-level metrics (request count/latency/CPU/memory, per revision) to
Cloud Monitoring for every deployed service with zero extra setup — between
the two, that covers what a self-managed Prometheus + Grafana stack would,
without running one.

```bash
python scripts/load_test.py http://localhost:8000 --requests 300 --concurrency 20
```

Local run against the full 3-service stack on this dev box: 300 requests,
20 concurrent, 100% success, p50 838ms / p95 1983ms / p99 2416ms. Those
numbers are dev-box artifacts, not a real capacity claim — a FAISS search
over a 600-item index and a LightGBM predict over 50 rows are each
sub-10ms operations in isolation, so multi-hundred-ms end-to-end latency
here points at request/threadpool/networking overhead specific to this
shared, multi-process Windows dev environment rather than the retrieval or
ranking logic itself. The real, citable p50/p95/p99 numbers are the ones
from `scripts/load_test.py` run against the actual deployed Cloud Run
gateway URL once `deploy/deploy.sh` has run — noted here instead of
papering over it, since "here's a number and here's why I don't trust it
yet" is more useful than a fake-confident one.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
pip install -e .               # installs `mise` + `services` so python -m / pytest work from anywhere

python -m mise.data_gen        # writes data/recipes.csv, users.csv, interactions.csv
python -m mise.train           # trains the two-tower model -> artifacts/model.pt + item_embeddings.npy
python -m mise.build_index     # builds artifacts/items.faiss from the item embeddings
python -m mise.rank_train      # trains the LightGBM ranker -> artifacts/ranker.txt
python -m mise.evaluate        # retrieval-only vs. retrieval+ranker vs. popularity/random baselines

uvicorn mise.api:app --reload  # http://localhost:8000/recommend?user_id=0
```

Feature store demo (separate, since it's an illustrative offline/online-split
capability rather than something wired into the live serving path yet):

```bash
python -m mise.popularity_gen               # writes feature_repo/data/*.parquet
(cd feature_repo && feast apply)            # registers entities + feature views, creates the sqlite online store
python -m mise.feature_store_demo           # point-in-time vs. naive-join comparison + online lookup
```

Run the smoke tests:

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
  popularity_gen.py time-varying recipe popularity + timestamped interactions for the feature store demo
  feature_store_demo.py  point-in-time-correct join vs. naive join, online store materialize + lookup
  job_queue.py      SQLite-backed durable job queue
  worker.py         polls the queue, processes catalog-refresh jobs
  api.py            monolithic FastAPI serving layer (simple reference implementation)
src/services/
  candidate_service.py  retrieval-only FastAPI app (two-tower + FAISS)
  ranking_service.py    reranking-only FastAPI app (LightGBM)
  gateway.py            composes the two services over real HTTP
feature_repo/
  feature_store.yaml  Feast project config (local provider, file offline store, sqlite online store)
  definitions.py      entities + feature views (recipe_popularity, user_profile)
  data/                generated parquet sources (gitignored, run popularity_gen.py to produce)
deploy/
  Dockerfile.candidate/ranking/gateway  one lean image per service
  requirements-candidate/ranking/gateway.txt  per-service dependency lists (no shared bloat)
  deploy.sh            builds via Cloud Build, deploys all 3 to Cloud Run
scripts/
  load_test.py         concurrent request load test against a running gateway
tests/
  test_pipeline.py    end-to-end: generate -> train -> index -> retrieve -> assert lift over popularity
  test_ranking.py     end-to-end: retrieval + ranker -> assert ranker doesn't regress retrieval order
  test_services.py    candidate-gen and ranking services work standalone
  test_job_queue.py   enqueue/claim/done/failed + dispatch mechanics
```
