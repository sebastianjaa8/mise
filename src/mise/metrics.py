"""Ranking metrics for retrieval evaluation."""
import numpy as np


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return float("nan")
    hits = len(set(retrieved[:k]) & relevant)
    return hits / min(k, len(relevant)) if len(relevant) < k else hits / k


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
