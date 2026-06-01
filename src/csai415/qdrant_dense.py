"""Pair B — Qdrant-backed dense backend for HybridRetriever. See D2_TASKS.md §D2-B2.

The blessed BOHB config uses ``metric="l2"``, but Qdrant ANN runs cosine for
fast candidate generation. The workaround:

- :meth:`QdrantDenseBackend.top_k` over-fetches ``k * OVERFETCH_MULT`` candidates
  from Qdrant under cosine ANN.
- :meth:`QdrantDenseBackend.scores_for` then fetches the candidate vectors
  via ``client.retrieve(..., with_vectors=True)`` and computes scores under the
  configured metric (L2 for blessed) in numpy.

So fusion sees true L2 scores. The only approximation is which candidates land
in the pool — with ``candidate_k=27`` and 2x over-fetch, the chance an L2-best
candidate falls outside cosine's top-54 is small. Expected NDCG@5 drift from
the D1 runcard is <0.5pp.

Point ID convention: ``corpus_idx`` (int, 0..n-1) matching the row order of
``chunks.parquet`` at seed time. Payload carries ``chunk_id`` + the metadata
``/search`` returns to clients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

QDRANT_COLLECTION = "chunks_bge384"
QDRANT_VECTOR_SIZE = 384
OVERFETCH_MULT = 2

Metric = Literal["cosine", "l2", "dot"]


class QdrantDenseBackend:
    """Dense backend backed by a Qdrant collection.

    Duck-typed to the interface ``HybridRetriever`` consumes:
    ``top_k(query_vec, k) -> [(corpus_idx, score), ...]`` and
    ``scores_for(query_vec, corpus_indices) -> np.ndarray``.
    """

    def __init__(
        self,
        client,
        collection: str = QDRANT_COLLECTION,
        metric: Metric = "l2",
        source_filter: Optional[str] = None,
    ):
        self.client = client
        self.collection = collection
        self.metric = metric
        self.source_filter = source_filter

        info = client.get_collection(collection)
        self.n_points = info.points_count
        if self.n_points == 0:
            raise RuntimeError(
                f"Qdrant collection {collection!r} has 0 points — run "
                f"seed_collection_from_parquet() before using this backend"
            )

    def _query_filter(self):
        if not self.source_filter:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=self.source_filter))]
        )

    def top_k(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top ``k * OVERFETCH_MULT`` candidates under cosine ANN."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.astype(np.float32).tolist(),
            limit=k * OVERFETCH_MULT,
            query_filter=self._query_filter(),
            with_payload=False,
            with_vectors=False,
        )
        return [(int(h.id), float(h.score)) for h in response.points]

    def scores_for(self, query_vec: np.ndarray, corpus_indices: np.ndarray) -> np.ndarray:
        """Fetch vectors for ``corpus_indices`` and score under ``self.metric``.

        ``HybridRetriever`` calls this during fusion to rescore the union of
        BM25 + dense candidates so the blessed L2 metric is honored even when
        candidate generation ran under cosine.
        """
        ids = [int(i) for i in corpus_indices]
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=ids,
            with_vectors=True,
            with_payload=False,
        )
        id_to_vec = {int(p.id): np.asarray(p.vector, dtype=np.float32) for p in points}
        try:
            vecs = np.stack([id_to_vec[i] for i in ids])
        except KeyError as exc:
            raise RuntimeError(
                f"Qdrant retrieve missing id {exc!s} — collection out of sync with parquet?"
            ) from exc

        q = query_vec.astype(np.float32)
        if self.metric in ("cosine", "dot"):
            return vecs @ q
        diffs = vecs - q
        return -np.linalg.norm(diffs, axis=1)  # l2: negate so higher = better


def _none_or_int(v) -> Optional[int]:
    """Coerce pandas null / NaN into Python None; Qdrant payload rejects NaN."""
    import math

    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def seed_collection_from_parquet(
    client,
    parquet_path: Path,
    collection: str = QDRANT_COLLECTION,
    batch_size: int = 256,
    recreate: bool = True,
) -> int:
    """Seed ``collection`` from ``chunks.parquet``. Returns the number of points upserted.

    With ``recreate=True`` the collection is dropped+rebuilt under
    ``Distance.COSINE`` and size 384 — the D2 contract. Point ID is the parquet
    row index so the corpus_idx convention used by ``HybridRetriever`` matches.
    Payload follows D2_TASKS.md (chunk_id, paper_id, source, page range, title,
    text) so ``/search`` can answer from a single store hit.
    """
    import pandas as pd
    from qdrant_client.models import Distance, PointStruct, VectorParams

    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    if "embedding" not in df.columns:
        raise RuntimeError(f"{parquet_path} has no 'embedding' column — corpus not embedded yet")

    if recreate:
        try:
            client.delete_collection(collection)
        except Exception:
            pass
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )

    n = len(df)
    for start in range(0, n, batch_size):
        sub = df.iloc[start : start + batch_size]
        points = []
        for i, row in sub.iterrows():
            vec = row["embedding"]
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            points.append(
                PointStruct(
                    id=int(i),
                    vector=list(vec),
                    payload={
                        "chunk_id": row["chunk_id"],
                        "paper_id": row["paper_id"],
                        "source": row["source"],
                        "page_start": _none_or_int(row.get("page_start")),
                        "page_end": _none_or_int(row.get("page_end")),
                        "title": (row.get("title") or "") if row.get("title") is not None else "",
                        "text": row["text"],
                    },
                )
            )
        client.upsert(collection_name=collection, points=points)
    return n
