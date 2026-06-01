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


class NumpyDenseBackend:
    """In-process dense backend — the D1 default. Holds the post-SVD,
    post-normalize corpus matrix and scores against it under ``metric``.

    Duck-typed to ``QdrantDenseBackend``: ``top_k(query_vec, k)`` and
    ``scores_for(query_vec, corpus_indices)``. ``HybridRetriever`` consumes
    either interchangeably.
    """

    def __init__(self, dense_matrix: np.ndarray, metric: Metric):
        self.dense_matrix = dense_matrix
        self.metric = metric

    def _all_scores(self, query_vec: np.ndarray) -> np.ndarray:
        if self.metric in ("cosine", "dot"):
            return self.dense_matrix @ query_vec
        diffs = self.dense_matrix - query_vec
        return -np.linalg.norm(diffs, axis=1)  # l2: negate so higher = better

    def top_k(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = self._all_scores(query_vec)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def scores_for(self, query_vec: np.ndarray, corpus_indices: np.ndarray) -> np.ndarray:
        return self._all_scores(query_vec)[corpus_indices]


class HybridRetriever:
    """BM25 + dense kNN with optional SVD on dense vectors. Owned by Pair B.

    The dense backend is pluggable (D2-B2): leave ``dense_backend=None`` to get
    the D1 numpy default, or inject a :class:`QdrantDenseBackend` for the
    FastAPI app. Backends must expose ``top_k`` (candidate generation) and
    ``scores_for`` (fusion-time rescoring under the configured metric).
    """

    def __init__(
        self,
        chunks_df: pd.DataFrame,
        config: RetrieverConfig,
        dense_backend=None,
    ):
        self.df = chunks_df.reset_index(drop=True)
        self.config = config
        self._dense_backend_override = dense_backend
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build BM25 over chunk text + dense backend (optional SVD, then normalize)."""
        tokenized = [t.lower().split() for t in self.df["text"]]
        self.bm25 = BM25Okapi(tokenized, k1=self.config.bm25_k1, b=self.config.bm25_b)

        # Optional SVD — fit on corpus now, .transform the query later in search()
        self.svd = None
        if self.config.svd_dim is not None:
            dense = np.array(self.df["embedding"].tolist(), dtype=np.float32)
            self.svd = TruncatedSVD(
                n_components=self.config.svd_dim,
                random_state=self.config.seed,
            )
            dense = self.svd.fit_transform(dense).astype(np.float32)
            if self.config.normalize:
                dense = normalize(dense).astype(np.float32)
            self.dense_backend = NumpyDenseBackend(dense, self.config.metric)
        elif self._dense_backend_override is not None:
            # Injected backend (typically Qdrant). SVD must be off because we
            # can't apply it to an external corpus we don't materialize here.
            if self.config.normalize:
                raise ValueError(
                    "config.normalize=True with an injected dense_backend is unsupported — "
                    "normalize the seeded vectors externally if needed"
                )
            self.dense_backend = self._dense_backend_override
        else:
            dense = np.array(self.df["embedding"].tolist(), dtype=np.float32)
            if self.config.normalize:
                dense = normalize(dense).astype(np.float32)
            self.dense_backend = NumpyDenseBackend(dense, self.config.metric)

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

    def _ranked_candidates(
        self, query: str, k: int, hybrid_weight: float | None = None
    ) -> list[tuple[str, float]]:
        """Core fusion. Returns the top-k ``(chunk_id, fused_score)`` pairs,
        sorted by descending fused score.

        ``search`` and ``search_with_scores`` are both thin wrappers over this,
        so the BM25/dense fusion math lives in exactly one place. The fused
        score is the per-query min-max-scaled weighted sum — comparable WITHIN
        a query's result list, not across queries.
        """
        w = hybrid_weight if hybrid_weight is not None else self.config.hybrid_weight
        c_k = self.config.candidate_k

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argpartition(-bm25_scores, c_k)[:c_k]

        q_vec = self._embed_query(query)
        dense_hits = self.dense_backend.top_k(q_vec, c_k)
        dense_top = np.asarray([i for i, _ in dense_hits], dtype=np.int64)

        # Union of candidates, min-max scale each backend's raw scores over the pool
        candidates = np.unique(np.concatenate([bm25_top, dense_top]))
        bm25_pool = bm25_scores[candidates]
        dense_pool = self.dense_backend.scores_for(q_vec, candidates)
        bm25_norm = (bm25_pool - bm25_pool.min()) / max(np.ptp(bm25_pool), 1e-12)
        dense_norm = (dense_pool - dense_pool.min()) / max(np.ptp(dense_pool), 1e-12)

        fused = w * dense_norm + (1 - w) * bm25_norm
        order = np.argsort(-fused)[:k]
        return [(self.df.iloc[candidates[i]]["chunk_id"], float(fused[i])) for i in order]

    def search(self, query: str, k: int, hybrid_weight: float | None = None) -> list[str]:
        """Top-k chunk_ids. hybrid_weight overrides config.hybrid_weight if given."""
        return [chunk_id for chunk_id, _ in self._ranked_candidates(query, k, hybrid_weight)]

    def search_with_scores(
        self, query: str, k: int, hybrid_weight: float | None = None
    ) -> list[tuple[str, float]]:
        """Top-k ``(chunk_id, fused_score)`` pairs — same ranking as ``search``,
        with the fused score exposed for the FastAPI ``/search`` response (D2-B1)."""
        return self._ranked_candidates(query, k, hybrid_weight)


def load_chunks(path: Path = CHUNKS_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)


def make_retriever_fn(retriever: HybridRetriever) -> Callable[[str, int, float], list[str]]:
    """Wrap a HybridRetriever instance into the contract signature."""
    def fn(query: str, k: int, hybrid_weight: float) -> list[str]:
        return retriever.search(query, k, hybrid_weight=hybrid_weight)
    return fn
