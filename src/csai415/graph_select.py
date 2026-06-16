"""D3-1 — Entity linking → subgraph selection. See briefs/D3_TASKS.md (Task D3-1).

Owner: Task D3-1. Given a query, decide which Neo4j nodes/paths seed the subgraph
the GraphRAG executor expands from.

**The CONTRACT is frozen at H0** — ``SubgraphResult`` and the ``select_subgraph``
signature. The body is an H0 stub the owner replaces *after* comparing the
approaches in the task doc (LLM entity extraction vs spaCy NER vs fuzzy match;
which of the 5 D2 Cypher templates to fire; 1-hop vs 2-hop breadth). When no graph
node matches, return a ``fallback`` result with empty ``paper_ids`` — the executor
degrades to pure vector/hybrid retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubgraphResult:
    """The subgraph chosen for a query.

    paper_ids is the contract the executor consumes: the set of papers whose chunks
    the graph-guided stage will filter/boost/expand toward. Empty ⇒ no graph hit.
    """

    seed_nodes: list[str] = field(default_factory=list)  # matched Author/Topic/Paper names
    paper_ids: list[str] = field(default_factory=list)   # papers in the selected subgraph
    cypher_used: str | None = None                       # which template fired (None if fallback)
    method: str = "fallback"                             # "fuzzy" | "ner" | "llm" | "fallback"


def select_subgraph(query: str, *, neo4j_driver=None, **kwargs) -> SubgraphResult:
    """Map ``query`` → graph nodes → a subgraph of relevant ``paper_ids``.

    H0 STUB: always returns an empty ``fallback`` so the executor runs without Neo4j.
    D3-1 owner implements real entity linking + Cypher routing and returns populated
    ``paper_ids`` / ``cypher_used`` / ``method``.
    """
    return SubgraphResult(method="fallback")
