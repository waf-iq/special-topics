"""Pair B — Hybrid retriever (BM25 + dense kNN). See MEMBER_BRIEF.md §6.B.

Contract:
    retriever_fn(query: str, k: int, hybrid_weight: float) -> list[str]
    hybrid_weight=1.0 -> pure dense; 0.0 -> pure BM25.
Both Pair B's AutoML and Pair C's online learner consume this signature.

Design decisions (from my AI session for D1):
* Dense ANN: brute-force numpy. 6,020 vectors x 384 dims is ~9 MB — sklearn
  NearestNeighbors or hnswlib add deps and complexity for no measurable win.
* SVD order: TruncatedSVD fit on corpus, .transform on query, then L2 normalize
  AFTER SVD. Normalizing before SVD leaves the reduced vectors with arbitrary
  magnitudes, so cosine/dot are no longer well-defined.
* BGE prefix: bge-small-en-v1.5 is asymmetric — query side gets the prefix
  "Represent this sentence for searching relevant passages: ", corpus side
  does not. Pair A embedded the corpus without it; we add it only at query
  time inside _embed_query.
* Fusion: weighted-sum of per-query min-max scaled BM25 + dense scores. RRF
  would sidestep BM25 unboundedness but throws away magnitude and doesn't
  honor the `hybrid_weight` contract that Pair B's Optuna and Pair C's online
  learner depend on.
* Candidate pool: union of top candidate_k from each backend BEFORE fusion;
  scaling happens over the union so both backends are normalized to the same
  reference frame per query.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")

Metric = Literal["cosine", "l2", "dot"]

# bge-small-en-v1.5 is asymmetric — query side gets a prefix, corpus side doesn't.
# Pair A embedded the corpus without it, so we add it here only for the query side.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class RetrieverConfig:
    """Captures every hyperparameter Optuna can search over.

    candidate_k = how many candidates to pull from EACH backend (BM25, dense)
    before fusion. The caller's `k` in .search(query, k, ...) is the final
    top-k returned. Splitting them avoids the NDCG@5 / candidate-pool confusion.
    """
    metric: Metric = "cosine"
    svd_dim: int | None = None         # None means no SVD
    normalize: bool = True
    hybrid_weight: float = 0.5
    candidate_k: int = 10              # pool size per backend before fusion
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    seed: int = 42



_embedder = None


def _get_embedder():
    """Cache the SBERT model at module level — loading per trial would kill runtime."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


class HybridRetriever:
    """BM25 + dense kNN with optional SVD on dense vectors. Owned by Pair B."""

    def __init__(self, chunks_df: pd.DataFrame, config: RetrieverConfig):
        self.df = chunks_df.reset_index(drop=True)
        self.config = config
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build BM25 over chunk text + dense matrix (optional SVD, then normalize)."""
        tokenized = [t.lower().split() for t in self.df["text"]]
        self.bm25 = BM25Okapi(tokenized, k1=self.config.bm25_k1, b=self.config.bm25_b)
        dense = np.array(self.df["embedding"].tolist(), dtype=np.float32)

        # Optional SVD — fit on corpus now, .transform the query later in search()
        self.svd = None
        if self.config.svd_dim is not None:
            self.svd = TruncatedSVD(
                n_components=self.config.svd_dim,
                random_state=self.config.seed,
            )
            dense = self.svd.fit_transform(dense).astype(np.float32)

        # Normalize AFTER SVD so cosine/dot are well-defined on the reduced vectors
        if self.config.normalize:
            dense = normalize(dense).astype(np.float32)

        self.dense_matrix = dense
        self.embedder = _get_embedder()

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query with the BGE prefix, then mirror corpus-side SVD + normalize."""
        text = BGE_QUERY_PREFIX + query
        vec = self.embedder.encode([text], convert_to_numpy=True).astype(np.float32)

        if self.svd is not None:
            vec = self.svd.transform(vec).astype(np.float32)
        if self.config.normalize:
            vec = normalize(vec).astype(np.float32)
        return vec[0]

    def _dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Score every corpus chunk under the configured metric (higher = better)."""
        if self.config.metric in ("cosine", "dot"):
            return self.dense_matrix @ query_vec
        # l2: distance — negate so higher = better matches the other metrics
        diffs = self.dense_matrix - query_vec
        return -np.linalg.norm(diffs, axis=1)

    def search(self, query: str, k: int, hybrid_weight: float | None = None) -> list[str]:
        """Top-k chunk_ids. hybrid_weight overrides config.hybrid_weight if given."""
        w = hybrid_weight if hybrid_weight is not None else self.config.hybrid_weight
        c_k = self.config.candidate_k

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argpartition(-bm25_scores, c_k)[:c_k]

        q_vec = self._embed_query(query)
        dense_scores = self._dense_scores(q_vec)
        dense_top = np.argpartition(-dense_scores, c_k)[:c_k]

        # Union of candidates, min-max scale each backend's raw scores over the pool
        candidates = np.unique(np.concatenate([bm25_top, dense_top]))
        bm25_pool = bm25_scores[candidates]
        dense_pool = dense_scores[candidates]
        bm25_norm = (bm25_pool - bm25_pool.min()) / max(np.ptp(bm25_pool), 1e-12)
        dense_norm = (dense_pool - dense_pool.min()) / max(np.ptp(dense_pool), 1e-12)

        fused = w * dense_norm + (1 - w) * bm25_norm
        order = np.argsort(-fused)[:k]
        return [self.df.iloc[candidates[i]]["chunk_id"] for i in order]


def load_chunks(path: Path = CHUNKS_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)


def make_retriever_fn(retriever: HybridRetriever) -> Callable[[str, int, float], list[str]]:
    """Wrap a HybridRetriever instance into the contract signature."""
    def fn(query: str, k: int, hybrid_weight: float) -> list[str]:
        return retriever.search(query, k, hybrid_weight=hybrid_weight)
    return fn
