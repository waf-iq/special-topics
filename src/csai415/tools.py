"""Thin tool layer for the GraphRAG executor + the deny-risky-tool-call safety demo.

See briefs/D3_TASKS.md (Task D3-2 exposes the tools; Task D3-7 hardens the guard).

The rubric wants a *fixed executor*, not a ReAct planner — so these are plain functions,
not an agent loop. They exist mainly so D3-7's "deny risky tool calls" mitigation has a
concrete seam to guard: ``cypher_query`` refuses write/destructive Cypher unless the
caller explicitly opts in.

**Frozen at H0:** ``RiskyToolCallDenied`` and ``is_write_cypher`` (pure, testable). The
I/O bodies (``vector_search``, ``cypher_query`` execution, ``read_pdf_page_range``) are
thin stubs the owners wire to the live backends.
"""

from __future__ import annotations

import re

# Clause keywords that mutate the graph. D3-7 owner tunes this set as part of comparing
# guard strategies (keyword allowlist vs a parser). ``CALL`` is intentionally excluded
# (read-only procedures use it); revisit if the executor never needs procedures.
WRITE_KEYWORDS: set[str] = {
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "FOREACH",
}


class RiskyToolCallDenied(RuntimeError):
    """Raised when a guarded tool refuses a dangerous call (D3-7 before/after evidence)."""


def is_write_cypher(cypher: str) -> bool:
    """True if the Cypher contains a mutating clause. Word-boundary match so 'created_at'
    as a property name doesn't trip 'CREATE'."""
    tokens = set(re.findall(r"[A-Za-z_]+", cypher.upper()))
    return bool(tokens & WRITE_KEYWORDS) or "LOAD CSV" in cypher.upper()


def cypher_query(driver, cypher: str, params: dict | None = None, *, allow_writes: bool = False) -> list[dict]:
    """Run a read-only Cypher query. Refuses writes unless ``allow_writes=True``.

    H0: the guard is real; the execution is a thin stub the owner wires to the Neo4j driver.
    """
    if not allow_writes and is_write_cypher(cypher):
        raise RiskyToolCallDenied(f"refused write/destructive Cypher: {cypher!r}")
    raise NotImplementedError("D3-2/D3-7 owner: execute read query via the Neo4j driver here")


def vector_search(query: str, k: int = 5, source: str | None = None) -> list[dict]:
    """Wrap the hybrid retriever as a tool. H0 stub — owner wires the executor's retriever."""
    raise NotImplementedError("D3-2 owner: route to HybridRetriever.search_with_scores here")


def read_pdf_page_range(paper_id: str, page_start: int, page_end: int) -> str:
    """Return the raw text for a page range of an arXiv PDF (citation verification, D3-4/D3-7).
    H0 stub — owner wires PyMuPDF over data/raw_pdfs/."""
    raise NotImplementedError("D3-4/D3-7 owner: read pages via PyMuPDF over data/raw_pdfs/")
