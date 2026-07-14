"""Each split-out service should work standalone (in-process TestClient, no
real network) — the gateway's real-HTTP composition is exercised manually
(see README) since spinning up two live uvicorn processes inside a test run
trades reliability for a marginal amount of extra coverage over this."""
from fastapi.testclient import TestClient

from services.candidate_service import app as candidate_app
from services.ranking_service import app as ranking_app


def test_candidate_service_returns_ranked_candidates():
    with TestClient(candidate_app) as client:
        assert client.get("/health").json()["status"] == "ok"

        resp = client.get("/candidates", params={"user_id": 0, "k": 5})
        assert resp.status_code == 200
        candidates = resp.json()["candidates"]
        assert len(candidates) == 5
        assert all("recipe_id" in c and "two_tower_score" in c for c in candidates)


def test_candidate_service_unknown_user_404s():
    with TestClient(candidate_app) as client:
        resp = client.get("/candidates", params={"user_id": 999999, "k": 5})
        assert resp.status_code == 404


def test_ranking_service_reorders_candidates():
    with TestClient(candidate_app) as client:
        candidates = client.get("/candidates", params={"user_id": 0, "k": 20}).json()["candidates"]

    with TestClient(ranking_app) as client:
        assert client.get("/health").json()["status"] == "ok"

        resp = client.post("/rerank", json={"user_id": 0, "candidates": candidates, "k": 5})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 5
        assert all("title" in r for r in results)
