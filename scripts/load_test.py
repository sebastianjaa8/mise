"""Load test: fire N requests at a running mise gateway (local or a
deployed Cloud Run URL) and report throughput + latency percentiles.

Usage: python scripts/load_test.py http://localhost:8000 --requests 500 --concurrency 20
"""
import argparse
import asyncio
import time

import httpx
import numpy as np


async def _one_request(client: httpx.AsyncClient, base_url: str, user_id: int, k: int):
    start = time.perf_counter()
    try:
        resp = await client.get(f"{base_url}/recommend", params={"user_id": user_id, "k": k})
        ok = resp.status_code == 200
    except httpx.HTTPError:
        ok = False
    return time.perf_counter() - start, ok


async def run(base_url: str, n_requests: int, concurrency: int, k: int, n_users: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(client, user_id):
        async with semaphore:
            return await _one_request(client, base_url, user_id, k)

    async with httpx.AsyncClient(timeout=10.0) as client:
        start = time.perf_counter()
        results = await asyncio.gather(*[bounded(client, i % n_users) for i in range(n_requests)])
        wall_clock = time.perf_counter() - start

    latencies_ms = np.array([r[0] for r in results]) * 1000
    successes = sum(r[1] for r in results)

    print(f"{n_requests} requests, concurrency={concurrency}, {wall_clock:.2f}s wall clock")
    print(f"throughput: {n_requests / wall_clock:.1f} req/s")
    print(f"success rate: {successes}/{n_requests} ({successes / n_requests:.1%})")
    print(f"latency  p50={np.percentile(latencies_ms, 50):.1f}ms  "
          f"p95={np.percentile(latencies_ms, 95):.1f}ms  "
          f"p99={np.percentile(latencies_ms, 99):.1f}ms  "
          f"max={latencies_ms.max():.1f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", help="e.g. http://localhost:8000 or a deployed Cloud Run gateway URL")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-users", type=int, default=400, help="cycles through user_id 0..n-1")
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.requests, args.concurrency, args.k, args.n_users))
