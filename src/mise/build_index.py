"""Build the FAISS ANN index from the offline-precomputed item embeddings.

Item embeddings change only when the catalog changes (new recipes, or a
periodic model refresh) — decoupling index build from request-time serving
is what makes sub-millisecond retrieval possible over a large catalog.
"""
import faiss
import numpy as np


def build(embeddings_path="artifacts/item_embeddings.npy", out_path="artifacts/items.faiss"):
    embeddings = np.load(embeddings_path).astype("float32")
    # embeddings are already L2-normalized by the item tower, so inner
    # product search is equivalent to cosine similarity ranking.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, out_path)
    print(f"built FAISS index: {index.ntotal} vectors, dim={embeddings.shape[1]} -> {out_path}")
    return index


if __name__ == "__main__":
    build()
