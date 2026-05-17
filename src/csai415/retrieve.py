"""Pair B — Hybrid retriever (BM25 + dense kNN). See MEMBER_BRIEF.md §6.B.

Contract:
    retriever_fn(query: str, k: int, hybrid_weight: float) -> list[str]
    hybrid_weight=1.0 -> pure dense; 0.0 -> pure BM25.
Both Pair B's AutoML and Pair C's online learner consume this signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")

Metric = Literal["cosine", "l2", "dot"]


@dataclass
class RetrieverConfig:
    """Captures every hyperparameter Optuna can search over."""
    metric: Metric = "cosine"
    svd_dim: int | None = None         # None means no SVD
    normalize: bool = True
    hybrid_weight: float = 0.5
    seed: int = 42


class HybridRetriever:
    """BM25 + dense kNN with optional SVD on dense vectors. Owned by Pair B."""

    def __init__(self, chunks_df: pd.DataFrame, config: RetrieverConfig):
        self.df = chunks_df
        self.config = config
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build BM25 index over chunk text and dense ANN index over embeddings."""
        raise NotImplementedError("Pair B — see §6.B question 3 (SVD + cosine normalization gotcha).")

    def search(self, query: str, k: int, hybrid_weight: float | None = None) -> list[str]:
        """Return top-k chunk_ids. hybrid_weight overrides config if given."""
        raise NotImplementedError("Pair B — see §6.B question 4 (RRF vs weighted-sum tradeoffs).")


def load_chunks(path: Path = CHUNKS_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)


def make_retriever_fn(retriever: HybridRetriever) -> Callable[[str, int, float], list[str]]:
    """Wrap a HybridRetriever instance into the contract signature."""
    def fn(query: str, k: int, hybrid_weight: float) -> list[str]:
        return retriever.search(query, k, hybrid_weight=hybrid_weight)
    return fn
