import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.pkl"
DOCS_PATH = DATA_DIR / "docs.json"


class VectorStore:
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        self._nn: NearestNeighbors | None = None
        self._embeddings: np.ndarray | None = None
        self._docs: List[Dict[str, Any]] = []

        self._load()

    def _load(self):
        if DOCS_PATH.exists():
            self._docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
        if INDEX_PATH.exists():
            with open(INDEX_PATH, "rb") as f:
                payload = pickle.load(f)
                self._embeddings = payload["embeddings"]
                self._nn = payload["nn"]

    def _persist(self):
        DOCS_PATH.write_text(json.dumps(self._docs, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._nn is not None and self._embeddings is not None:
            with open(INDEX_PATH, "wb") as f:
                pickle.dump({"nn": self._nn, "embeddings": self._embeddings}, f)

    def add(self, docs: List[Dict[str, Any]], embeddings: List[List[float]]):
        if not docs:
            return

        emb = np.array(embeddings, dtype=np.float32)
        if self._embeddings is None:
            self._embeddings = emb
            self._docs = docs
        else:
            self._embeddings = np.vstack([self._embeddings, emb])
            self._docs.extend(docs)

        # cosine metric via normalized vectors
        X = self._normalize(self._embeddings)
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._nn.fit(X)

        self._persist()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self._nn is None or self._embeddings is None or not self._docs:
            return []

        q = np.array([query_embedding], dtype=np.float32)
        Xq = self._normalize(q)
        X = self._normalize(self._embeddings)

        distances, indices = self._nn.kneighbors(Xq, n_neighbors=min(top_k, len(self._docs)))
        results: List[Tuple[Dict[str, Any], float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            score = float(1.0 - dist)  # cosine similarity
            results.append((self._docs[int(idx)], score))
        return results

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return x / norm
