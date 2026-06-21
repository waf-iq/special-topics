"""Task 1 (D3 GraphRAG, 8%) — Graph-guided retrieval. See briefs/D3_D4_TASKS.md.

Owner: Yousef Alsakkaf. Given a query, decide which Neo4j nodes seed the subgraph and
return the ``paper_ids`` the GraphRAG executor filters/boosts/expands toward.

Pipeline
--------
    query
      → entity linking   (query spans → Author / Topic graph nodes)   [fuzzy | spacy | llm]
      → Cypher routing   (which D2 template fires, given the linked node types)
      → subgraph         (SubgraphResult.paper_ids the executor consumes)

The **contract is frozen at H0** — ``SubgraphResult`` (its four core fields) and the
``select_subgraph(query, *, neo4j_driver=None) -> SubgraphResult`` signature. Extra
reporting fields on ``SubgraphResult`` carry defaults, so the executor and the contract
test see the same shape. When nothing links (or no driver is given) we return a
``method="fallback"`` result with empty ``paper_ids`` and the executor degrades to pure
vector/hybrid retrieval.

Three things this module compares (the Task-1 deliverable tables live in
``notebooks``/``reports``; the implementations they exercise are here):

1. **Entity linking** — ``link_fuzzy`` (rapidfuzz, offline, the shipped default) vs
   ``link_spacy`` (PERSON NER → author link) vs ``link_llm`` (the env-configured SLM/Groq
   backend extracts entities, then we link them to real nodes). All three feed one router.
2. **Cypher routing** — author → template 01; topic → template 04; author∩topic →
   intersection; collaboration intent → template 02; two topics → template 05. The corpus
   is all year 2026, so template 03 (year window) is degenerate and intentionally unused.
3. **Guidance policy** — ``apply_guidance`` implements graph-as-**filter** / **booster** /
   **expansion** so the integrator can wire any policy into the executor without editing it.

Heavy deps (rapidfuzz / spacy / openai) are imported lazily inside the functions that need
them, so ``import csai415.graph_select`` stays cheap and the no-dep contract test stays green.
"""

from __future__ import annotations

import os
import re
import time
import weakref
from dataclasses import dataclass, field

# --- Topic vocabulary ------------------------------------------------------------------
# Natural-language phrase → arXiv category code. Only codes actually present in the seeded
# graph end up linked (we intersect against the live Topic set), so listing extra synonyms
# is harmless. Keys are matched case-insensitively on word boundaries.
TOPIC_SYNONYMS: dict[str, str] = {
    "computational linguistics": "cs.CL",
    "natural language processing": "cs.CL",
    "nlp": "cs.CL",
    "language model": "cs.CL",
    "language models": "cs.CL",
    "text": "cs.CL",
    "computer vision": "cs.CV",
    "vision": "cs.CV",
    "image": "cs.CV",
    "visual": "cs.CV",
    "machine learning": "cs.LG",
    "deep learning": "cs.LG",
    "artificial intelligence": "cs.AI",
    "agent": "cs.AI",
    "agents": "cs.AI",
    "security": "cs.CR",
    "cryptography": "cs.CR",
    "adversarial": "cs.CR",
    "information retrieval": "cs.IR",
    "retrieval": "cs.IR",
    "search": "cs.IR",
    "recommendation": "cs.IR",
    "software engineering": "cs.SE",
    "code": "cs.SE",
    "human-computer interaction": "cs.HC",
    "hci": "cs.HC",
    "game theory": "cs.GT",
    "multimedia": "cs.MM",
    "speech": "eess.AS",
    "audio": "eess.AS",
    "computers and society": "cs.CY",
    "society": "cs.CY",
    "materials science": "cond-mat.mtrl-sci",
}

# Question words / common capitalized openers that must never be taken for an author span.
_STOP_SPANS = {
    "What", "Who", "Which", "When", "Where", "Why", "How", "Is", "Are", "Do", "Does",
    "Can", "Could", "Should", "The", "A", "An", "This", "That", "These", "Those",
}

_LINK_METHODS = ("fuzzy", "spacy", "llm")


# --- Data model ------------------------------------------------------------------------
@dataclass
class LinkedEntity:
    """One query span resolved to a graph node."""

    name: str          # canonical graph value: an Author.name or a Topic.name (category code)
    label: str         # "Author" | "Topic"
    score: float       # link confidence in [0, 1]
    span: str = ""     # the query substring that produced the match


@dataclass
class SubgraphResult:
    """The subgraph chosen for a query.

    ``paper_ids`` is the contract the executor consumes: papers whose chunks the graph
    stage filters/boosts/expands toward. Empty ⇒ no graph hit (fallback). The four core
    fields are the frozen H0 contract; the rest are reporting extras with defaults.
    """

    seed_nodes: list[str] = field(default_factory=list)   # matched Author/Topic names
    paper_ids: list[str] = field(default_factory=list)    # papers in the selected subgraph
    cypher_used: str | None = None                        # template id that fired (None ⇒ fallback)
    method: str = "fallback"                              # "fuzzy" | "spacy" | "llm" | "fallback"
    # --- reporting extras (do not affect the executor contract) ---
    linked: list[LinkedEntity] = field(default_factory=list)
    latency_ms: float = 0.0


class GraphUnavailable(RuntimeError):
    """Raised internally when the graph cannot be queried; callers degrade to fallback."""


class BackendUnavailable(RuntimeError):
    """Raised by ``link_llm`` when no CSAI415_ANSWERER backend is configured."""


# --- Graph index (cached author/topic vocabulary) --------------------------------------
class GraphIndex:
    """In-memory vocabulary pulled once from the graph: author names + topic codes.

    Entity linking matches query spans against these lists rather than hitting Neo4j per
    span, so a query costs one fuzzy sweep (~hundreds of names) + one Cypher round-trip.
    """

    def __init__(self, authors: list[str], topics: list[str]):
        self.authors = list(authors)
        self.topics = list(topics)
        self._topics_lower = {t.lower(): t for t in self.topics}

    @classmethod
    def from_driver(cls, driver) -> "GraphIndex":
        with driver.session() as s:
            authors = [r["name"] for r in s.run("MATCH (a:Author) RETURN a.name AS name")]
            topics = [r["name"] for r in s.run("MATCH (t:Topic) RETURN t.name AS name")]
        return cls(authors, topics)


# Cache GraphIndex per driver object so repeated calls (eval loops, the executor) reuse it.
# WeakKeyDictionary (not id(driver)→idx): entries auto-evict when a driver is garbage
# collected, so a reused memory address can never return another driver's stale vocabulary.
_INDEX_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def get_index(driver) -> GraphIndex:
    idx = _INDEX_CACHE.get(driver)
    if idx is None:
        idx = GraphIndex.from_driver(driver)
        _INDEX_CACHE[driver] = idx
    return idx


def clear_index_cache() -> None:
    """Drop the cached vocabularies (call after reseeding the graph in a long session)."""
    _INDEX_CACHE.clear()


# --- Span extraction -------------------------------------------------------------------
def _name_spans(query: str) -> list[str]:
    """Candidate author-name spans: runs of Capitalized tokens, sentence openers dropped.

    Author names are multi-token Capitalized runs ("Yann LeCun", "Jianing Hao"). We also
    keep single Capitalized tokens as weak candidates — the fuzzy threshold rejects the
    ones that are not real names, while a lone surname can still link.
    """
    runs = re.findall(r"\b([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+)*)\b", query)
    spans: list[str] = []
    for run in runs:
        toks = run.split()
        # Drop a leading question/stop word ("What has Yann ..." → "Yann ...").
        if toks and toks[0] in _STOP_SPANS:
            toks = toks[1:]
        if not toks:
            continue
        cleaned = " ".join(toks)
        spans.append(cleaned)
    return spans


# --- Topic linking (shared by every method) --------------------------------------------
def _link_topics(text: str, index: GraphIndex) -> list[LinkedEntity]:
    """Link a query to Topic nodes via explicit category codes + the synonym table."""
    text_l = text.lower()
    best: dict[str, LinkedEntity] = {}

    def consider(code: str, score: float, span: str) -> None:
        if code not in index.topics:  # only link to topics that exist in the live graph
            return
        prev = best.get(code)
        if prev is None or score > prev.score:
            best[code] = LinkedEntity(code, "Topic", score, span)

    # Explicit category code, e.g. "cs.CL" written verbatim in the query.
    for code in index.topics:
        if code.lower() in text_l:
            consider(code, 1.0, code)
    # Synonym phrases on word boundaries.
    for phrase, code in TOPIC_SYNONYMS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text_l):
            consider(code, 0.9, phrase)
    return list(best.values())


# --- Author linking, per method --------------------------------------------------------
def _fuzzy_link_authors(spans: list[str], index: GraphIndex, threshold: int) -> list[LinkedEntity]:
    """Fuzzy-match candidate spans against the author vocabulary (rapidfuzz WRatio)."""
    if not spans or not index.authors:
        return []
    from rapidfuzz import fuzz, process

    out: dict[str, LinkedEntity] = {}
    for span in spans:
        match = process.extractOne(span, index.authors, scorer=fuzz.WRatio)
        if match and match[1] >= threshold:
            name, score = match[0], match[1]
            prev = out.get(name)
            if prev is None or score > prev.score * 100:
                out[name] = LinkedEntity(name, "Author", score / 100.0, span)
    return list(out.values())


def link_fuzzy(query: str, index: GraphIndex, *, author_threshold: int = 88) -> list[LinkedEntity]:
    """Offline baseline + shipped default: rapidfuzz on Capitalized spans + topic synonyms."""
    authors = _fuzzy_link_authors(_name_spans(query), index, author_threshold)
    return authors + _link_topics(query, index)


_SPACY_NLP = None


def _get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return _SPACY_NLP


def link_spacy(query: str, index: GraphIndex, *, author_threshold: int = 85) -> list[LinkedEntity]:
    """spaCy PERSON entities → fuzzy author link; topics via the shared synonym table."""
    doc = _get_spacy()(query)
    person_spans = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    authors = _fuzzy_link_authors(person_spans, index, author_threshold)
    return authors + _link_topics(query, index)


_LLM_PROMPT = (
    "Extract named entities from the user's question about a corpus of arXiv CS papers. "
    "Return ONLY a compact JSON object with two keys: "
    '"authors" (list of person names mentioned) and '
    '"topics" (list of research areas mentioned, e.g. "natural language processing"). '
    "Use [] when none are present. No prose, JSON only."
)


def link_llm(query: str, index: GraphIndex, *, author_threshold: int = 85) -> list[LinkedEntity]:
    """LLM-NER: the configured backend extracts entity strings; we link them to real nodes.

    Reuses ``answer.resolve_backend`` (the OpenAI-compatible Ollama/Groq client) so the
    same ``CSAI415_ANSWERER`` env drives linking and answering. The model only proposes raw
    strings — they are still fuzzy-linked to the actual graph vocabulary, so an LLM that
    hallucinates a name simply fails to link rather than inventing a node.
    """
    from csai415.answer import resolve_backend

    backend = resolve_backend()
    if backend is None:
        raise BackendUnavailable("CSAI415_ANSWERER is not set — no LLM backend for link_llm")
    client, model, _kind = backend

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _LLM_PROMPT},
                  {"role": "user", "content": query}],
        temperature=0.0,
        max_tokens=200,
    )
    raw = (resp.choices[0].message.content or "").strip()
    authors_raw, topics_raw = _parse_llm_entities(raw)

    authors = _fuzzy_link_authors(authors_raw, index, author_threshold)
    # Topic strings: run each back through the synonym/code matcher.
    topics: dict[str, LinkedEntity] = {}
    for t in topics_raw + [query]:  # include the query so explicit codes still catch
        for e in _link_topics(t, index):
            prev = topics.get(e.name)
            if prev is None or e.score > prev.score:
                topics[e.name] = e
    return authors + list(topics.values())


def _parse_llm_entities(raw: str) -> tuple[list[str], list[str]]:
    """Tolerant JSON-ish parse of the LLM's entity reply → (authors, topics)."""
    import json

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return [], []
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return [], []
    authors = [str(x).strip() for x in (obj.get("authors") or []) if str(x).strip()]
    topics = [str(x).strip() for x in (obj.get("topics") or []) if str(x).strip()]
    return authors, topics


_LINKERS = {"fuzzy": link_fuzzy, "spacy": link_spacy, "llm": link_llm}


def link_entities(query: str, index: GraphIndex, method: str = "fuzzy") -> list[LinkedEntity]:
    """Dispatch to a linker by name. ``method`` ∈ {"fuzzy", "spacy", "llm"}."""
    try:
        linker = _LINKERS[method]
    except KeyError:
        raise ValueError(f"unknown link method {method!r}; expected one of {_LINK_METHODS}")
    return linker(query, index)


# --- Cypher routing --------------------------------------------------------------------
# Parameterized versions of the D2 templates in cypher/. Each returns paper_ids for the
# selected subgraph. ``cypher_used`` records which template fired (the routing comparison).
# No trailing \b: "collaborat" must still match "collaborate"/"collaboration" (a \b right
# after the stem fails, since the next char is a word char).
_COLLAB_RE = re.compile(r"co-?authors?|collaborat\w*|works?\s+with|work(?:ed|ing)\s+with|team(?:s|ed)?\s+up", re.I)


def _run_paper_query(driver, cypher: str, **params) -> list[str]:
    with driver.session() as s:
        return [r["pid"] for r in s.run(cypher, **params)]


def route_subgraph(
    linked: list[LinkedEntity], driver, query: str, *, max_papers: int = 60
) -> tuple[list[str], str | None, list[str]]:
    """Pick the Cypher template from the linked node types; return (paper_ids, template, seeds).

    Priority: author∩topic (most precise) → collaboration → two-topic intersection →
    author → topic. Returns ([], None, []) when nothing routes (caller → fallback).
    """
    authors = [e.name for e in linked if e.label == "Author"]
    topics = [e.name for e in linked if e.label == "Topic"]

    # author ∩ topic — the author's papers within that topic (tightest non-empty subgraph).
    if authors and topics:
        pids = _run_paper_query(
            driver,
            "MATCH (a:Author)-[:WROTE]->(p:Paper)-[:ABOUT]->(t:Topic) "
            "WHERE a.name IN $names AND t.name IN $codes "
            "RETURN DISTINCT p.paper_id AS pid LIMIT $lim",
            names=authors, codes=topics, lim=max_papers,
        )
        if pids:
            return pids, "01+04_author_topic_intersect", authors + topics
        # fall through: intersection empty → try the broader author route below.

    # collaboration intent + an author → template 02 (author's papers + co-authors' papers).
    if authors and _COLLAB_RE.search(query):
        pids = _run_paper_query(
            driver,
            "MATCH (a:Author)-[:WROTE]->(p:Paper) WHERE a.name IN $names "
            "OPTIONAL MATCH (p)<-[:WROTE]-(co:Author)-[:WROTE]->(cp:Paper) "
            "WITH collect(DISTINCT p) + collect(DISTINCT cp) AS papers "  # two aggregations, no grouping key
            "UNWIND papers AS x RETURN DISTINCT x.paper_id AS pid LIMIT $lim",
            names=authors, lim=max_papers,
        )
        if pids:
            return pids, "02_top_coauthors", authors

    # two distinct topics → template 05 (papers about either; the cross-topic neighborhood).
    if len(topics) >= 2:
        pids = _run_paper_query(
            driver,
            "MATCH (p:Paper)-[:ABOUT]->(t:Topic) WHERE t.name IN $codes "
            "RETURN DISTINCT p.paper_id AS pid LIMIT $lim",
            codes=topics, lim=max_papers,
        )
        if pids:
            return pids, "05_authors_on_both_topics", topics

    # single author → template 01 (papers by author).
    if authors:
        pids = _run_paper_query(
            driver,
            "MATCH (a:Author)-[:WROTE]->(p:Paper) WHERE a.name IN $names "
            "RETURN DISTINCT p.paper_id AS pid LIMIT $lim",
            names=authors, lim=max_papers,
        )
        if pids:
            return pids, "01_papers_by_author", authors

    # single topic → template 04 (papers about topic).
    if topics:
        pids = _run_paper_query(
            driver,
            "MATCH (p:Paper)-[:ABOUT]->(t:Topic) WHERE t.name IN $codes "
            "RETURN DISTINCT p.paper_id AS pid LIMIT $lim",
            codes=topics, lim=max_papers,
        )
        if pids:
            return pids, "04_papers_and_authors_by_topic", topics

    return [], None, []


# --- Public contract -------------------------------------------------------------------
def select_subgraph(
    query: str,
    *,
    neo4j_driver=None,
    method: str = "auto",
    max_papers: int = 60,
    **kwargs,
) -> SubgraphResult:
    """Map ``query`` → graph nodes → a subgraph of relevant ``paper_ids``.

    ``method`` selects the linker: "fuzzy" | "spacy" | "llm", or "auto" (the shipped
    default → fuzzy: offline, fast, deterministic; overridable via ``CSAI415_GRAPH_LINK``).
    Returns a ``method="fallback"`` result with empty ``paper_ids`` whenever there is no
    driver, no link, or the graph errors — so the executor always degrades gracefully.
    """
    t0 = time.perf_counter()

    def _ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    # No linker ever produced output (no driver / linking error) → a true fallback.
    def fallback() -> SubgraphResult:
        return SubgraphResult(method="fallback", latency_ms=_ms())

    # Linking succeeded but the graph yielded no usable subgraph (no papers, or a query
    # error). Report the linker that actually ran — empty paper_ids still degrades the
    # executor to vector/hybrid, but the method field stays honest about what was attempted.
    def linked_no_hit(linked) -> SubgraphResult:
        return SubgraphResult(
            seed_nodes=[e.name for e in linked], method=chosen, linked=linked, latency_ms=_ms())

    if neo4j_driver is None:
        return fallback()

    chosen = method
    if chosen == "auto":
        chosen = (os.getenv("CSAI415_GRAPH_LINK") or "fuzzy").strip().lower()

    try:
        index = get_index(neo4j_driver)
        linked = link_entities(query, index, chosen)
    except BackendUnavailable:
        # LLM linker requested but no backend → degrade to the offline fuzzy linker.
        chosen = "fuzzy"
        try:
            index = get_index(neo4j_driver)
            linked = link_entities(query, index, "fuzzy")
        except Exception:
            return fallback()
    except Exception:
        return fallback()

    if not linked:
        return fallback()

    try:
        paper_ids, cypher_used, seeds = route_subgraph(
            linked, neo4j_driver, query, max_papers=max_papers
        )
    except Exception:
        return linked_no_hit(linked)  # linked OK, graph errored during routing

    if not paper_ids:
        return linked_no_hit(linked)  # linked OK, but the subgraph is empty

    return SubgraphResult(
        seed_nodes=seeds,
        paper_ids=paper_ids,
        cypher_used=cypher_used,
        method=chosen,
        linked=linked,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


# --- Guidance policy: how the subgraph reshapes retrieval -------------------------------
def apply_guidance(
    ranked_ids: list[str],
    subgraph: SubgraphResult,
    *,
    paper_of,
    policy: str = "filter",
    scores: dict[str, float] | None = None,
    chunks_by_paper: dict[str, list[str]] | None = None,
    boost: float = 0.25,
    max_expand: int = 10,
) -> list[str]:
    """Reshape a candidate ranking using the subgraph. Returns a reordered ``chunk_id`` list.

    Policies (the Task-1 comparison — graph-as-…):
      * ``"filter"``   keep only candidates whose paper is in the subgraph; if that empties
                       the pool, return the original ranking (graceful no-hit, never worse
                       than vector-only).
      * ``"booster"``  additively boost in-subgraph candidates' scores and re-sort — softer
                       than filtering, protects recall (needs ``scores``; falls back to a
                       rank-based proxy when not given).
      * ``"expansion"`` keep the ranking, then append chunks from subgraph papers that
                        vector search missed (needs ``chunks_by_paper``), capped at
                        ``max_expand`` — graph-first recall.

    ``paper_of(chunk_id) -> paper_id`` maps a candidate to its paper. With an empty/fallback
    subgraph every policy is a no-op (returns ``ranked_ids`` unchanged).
    """
    wanted = set(subgraph.paper_ids)
    if not wanted:
        return list(ranked_ids)

    if policy == "filter":
        kept = [cid for cid in ranked_ids if paper_of(cid) in wanted]
        return kept if kept else list(ranked_ids)

    if policy == "booster":
        n = len(ranked_ids)
        # Rank-based proxy score when explicit scores aren't supplied: 1.0 .. ~0 over the pool.
        base = scores or {cid: (n - i) / n for i, cid in enumerate(ranked_ids)}
        adjusted = {
            cid: base.get(cid, 0.0) + (boost if paper_of(cid) in wanted else 0.0)
            for cid in ranked_ids
        }
        return sorted(ranked_ids, key=lambda cid: adjusted[cid], reverse=True)

    if policy == "expansion":
        out = list(ranked_ids)
        present = set(out)
        if chunks_by_paper:
            added = 0
            for pid in subgraph.paper_ids:
                for cid in chunks_by_paper.get(pid, []):
                    if added >= max_expand:
                        break
                    if cid not in present:
                        out.append(cid)
                        present.add(cid)
                        added += 1
                if added >= max_expand:
                    break
        return out

    raise ValueError(f"unknown guidance policy {policy!r}; expected filter|booster|expansion")
