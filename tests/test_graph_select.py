"""Unit tests for D3 Task-1 graph-guided retrieval (src/csai415/graph_select.py).

No Docker / Neo4j / models: entity linking runs against a synthetic ``GraphIndex`` and
Cypher routing runs against a fake driver that answers by parameter shape. This pins the
linking + routing + guidance logic so refactors can't silently regress it. The live-graph
behaviour is exercised separately by ``check_graph.py`` and the Task-1 eval notebook.
"""

from __future__ import annotations

import pytest

from csai415.graph_select import (
    BackendUnavailable,
    GraphIndex,
    SubgraphResult,
    _COLLAB_RE,
    _fuzzy_link_authors,
    _link_topics,
    _name_spans,
    _parse_llm_entities,
    apply_guidance,
    clear_index_cache,
    get_index,
    link_entities,
    link_fuzzy,
    link_llm,
    link_spacy,
    route_subgraph,
    select_subgraph,
)

AUTHORS = ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Yann LeCun"]
TOPICS = ["cs.CL", "cs.CV", "cs.IR", "cs.LG"]


@pytest.fixture
def index() -> GraphIndex:
    return GraphIndex(AUTHORS, TOPICS)


# --- Fake Neo4j ------------------------------------------------------------------------
# A tiny graph; the fake session answers vocabulary + routing queries by parameter shape,
# mirroring how route_subgraph parameterizes each template (names / codes / co:Author).
_AUTHOR_PAPERS = {"Ada Lovelace": ["pA1", "pA2"], "Alan Turing": ["pT1"], "Yann LeCun": ["pY1"]}
_TOPIC_PAPERS = {"cs.CL": ["pA1", "pT1"], "cs.CV": ["pA2", "pY1"], "cs.IR": ["pY1"]}
_COAUTHOR_PAPERS = {"Ada Lovelace": ["pA1", "pA2", "pCo1"]}  # Ada + a coauthor's other paper


class _FakeSession:
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **params):
        if "RETURN a.name AS name" in cypher:
            return [{"name": n} for n in self.graph["authors"]]
        if "RETURN t.name AS name" in cypher:
            return [{"name": t} for t in self.graph["topics"]]
        return [{"pid": p} for p in self._route(cypher, params)]

    def _route(self, cypher, params):
        names, codes = params.get("names"), params.get("codes")
        if names and codes:  # author ∩ topic
            authors = {p for n in names for p in _AUTHOR_PAPERS.get(n, [])}
            topics = {p for c in codes for p in _TOPIC_PAPERS.get(c, [])}
            return sorted(authors & topics)
        if names and "co:Author" in cypher:  # collaboration expansion
            out: list[str] = []
            for n in names:
                out += _COAUTHOR_PAPERS.get(n, _AUTHOR_PAPERS.get(n, []))
            return sorted(set(out))
        if names:  # papers by author
            return sorted({p for n in names for p in _AUTHOR_PAPERS.get(n, [])})
        if codes:  # papers by topic(s)
            return sorted({p for c in codes for p in _TOPIC_PAPERS.get(c, [])})
        return []


class _FakeDriver:
    def __init__(self):
        self.graph = {"authors": AUTHORS, "topics": TOPICS}

    def session(self):
        return _FakeSession(self.graph)


@pytest.fixture
def driver():
    clear_index_cache()  # don't leak a cached index across tests
    return _FakeDriver()


# --- Span extraction -------------------------------------------------------------------
def test_name_spans_extracts_capitalized_runs():
    assert "Yann LeCun" in _name_spans("papers by Yann LeCun")


def test_name_spans_drops_leading_question_word():
    # "What has Grace Hopper ..." → the opener is stripped, the name survives.
    spans = _name_spans("What has Grace Hopper published?")
    assert any(s == "Grace Hopper" for s in spans)
    assert "What" not in spans


# --- Topic linking ---------------------------------------------------------------------
def test_link_topics_explicit_code(index):
    hits = _link_topics("recent work in cs.CV models", index)
    assert [h.name for h in hits] == ["cs.CV"]
    assert hits[0].score == 1.0


def test_link_topics_synonym(index):
    hits = _link_topics("a survey of natural language processing", index)
    assert any(h.name == "cs.CL" for h in hits)


def test_link_topics_ignores_absent_category(index):
    # "cs.SE" is not in this index's topic set → no link.
    assert _link_topics("software engineering tooling", index) == []


# --- Author linking --------------------------------------------------------------------
def test_fuzzy_links_exact_author(index):
    hits = _fuzzy_link_authors(["Ada Lovelace"], index, threshold=88)
    assert [h.name for h in hits] == ["Ada Lovelace"]


def test_fuzzy_tolerates_minor_typo(index):
    hits = _fuzzy_link_authors(["Ada Lovlace"], index, threshold=85)  # missing 'e'
    assert hits and hits[0].name == "Ada Lovelace"


def test_fuzzy_rejects_non_author(index):
    assert _fuzzy_link_authors(["Quantum Chromodynamics"], index, threshold=88) == []


def test_link_fuzzy_combines_author_and_topic(index):
    hits = link_fuzzy("Did Yann LeCun work on cs.CV?", index)
    labels = {h.label for h in hits}
    assert labels == {"Author", "Topic"}


def test_link_entities_unknown_method(index):
    with pytest.raises(ValueError):
        link_entities("x", index, "bogus")


# --- Collaboration intent regex --------------------------------------------------------
@pytest.mark.parametrize("q", [
    "Who does Ada collaborate with?",
    "papers co-authored by Turing",
    "researchers who work with Hopper",
    "the coauthors of LeCun",
])
def test_collab_regex_matches(q):
    assert _COLLAB_RE.search(q)


@pytest.mark.parametrize("q", ["What did Ada Lovelace write?", "papers about cs.CV"])
def test_collab_regex_non_match(q):
    assert not _COLLAB_RE.search(q)


# --- Cypher routing --------------------------------------------------------------------
def test_route_author_topic_intersection(index, driver):
    linked = link_fuzzy("Ada Lovelace papers in cs.CV", index)
    pids, cypher, seeds = route_subgraph(linked, driver, "Ada Lovelace papers in cs.CV")
    assert cypher == "01+04_author_topic_intersect"
    assert pids == ["pA2"]  # Ada wrote pA1,pA2; cs.CV has pA2,pY1 → ∩ = pA2


def test_route_single_author(index, driver):
    linked = link_fuzzy("What did Ada Lovelace publish?", index)
    pids, cypher, _ = route_subgraph(linked, driver, "What did Ada Lovelace publish?")
    assert cypher == "01_papers_by_author"
    assert set(pids) == {"pA1", "pA2"}


def test_route_collaboration(index, driver):
    q = "Who does Ada Lovelace collaborate with?"
    pids, cypher, _ = route_subgraph(link_fuzzy(q, index), driver, q)
    assert cypher == "02_top_coauthors"
    assert "pCo1" in pids  # picked up a coauthor's other paper


def test_route_two_topics(index, driver):
    q = "work bridging cs.CL and cs.IR"
    pids, cypher, _ = route_subgraph(link_fuzzy(q, index), driver, q)
    assert cypher == "05_authors_on_both_topics"
    assert set(pids) == {"pA1", "pT1", "pY1"}  # union of both topics


def test_route_single_topic(index, driver):
    q = "papers about cs.CV"
    pids, cypher, _ = route_subgraph(link_fuzzy(q, index), driver, q)
    assert cypher == "04_papers_and_authors_by_topic"
    assert set(pids) == {"pA2", "pY1"}


def test_route_no_link_returns_empty(index, driver):
    assert route_subgraph([], driver, "the capital of France") == ([], None, [])


# --- select_subgraph end-to-end (fake driver) ------------------------------------------
def test_select_subgraph_none_driver_is_fallback():
    res = select_subgraph("anything", neo4j_driver=None)
    assert isinstance(res, SubgraphResult)
    assert res.method == "fallback" and res.paper_ids == [] and res.cypher_used is None


def test_select_subgraph_author_hit(driver):
    res = select_subgraph("What did Ada Lovelace write?", neo4j_driver=driver, method="fuzzy")
    assert res.method == "fuzzy"
    assert res.cypher_used == "01_papers_by_author"
    assert set(res.paper_ids) == {"pA1", "pA2"}
    assert res.latency_ms >= 0


def test_select_subgraph_no_graph_hit_is_graceful(driver):
    res = select_subgraph("the capital of France", neo4j_driver=driver, method="fuzzy")
    assert res.method == "fallback" and res.paper_ids == []


def test_select_subgraph_llm_degrades_to_fuzzy_without_backend(driver, monkeypatch):
    # No CSAI415_ANSWERER → link_llm raises BackendUnavailable → select_subgraph retries fuzzy.
    monkeypatch.delenv("CSAI415_ANSWERER", raising=False)
    res = select_subgraph("papers by Ada Lovelace", neo4j_driver=driver, method="llm")
    assert res.method == "fuzzy" and set(res.paper_ids) == {"pA1", "pA2"}


def test_link_llm_raises_without_backend(index, monkeypatch):
    monkeypatch.delenv("CSAI415_ANSWERER", raising=False)
    with pytest.raises(BackendUnavailable):
        link_llm("papers by Ada", index)


# --- LLM entity parsing ----------------------------------------------------------------
def test_parse_llm_entities_clean_json():
    a, t = _parse_llm_entities('{"authors": ["Ada Lovelace"], "topics": ["vision"]}')
    assert a == ["Ada Lovelace"] and t == ["vision"]


def test_parse_llm_entities_embedded_json_with_prose():
    a, t = _parse_llm_entities('Sure! {"authors": ["Turing"], "topics": []} hope that helps')
    assert a == ["Turing"] and t == []


def test_parse_llm_entities_garbage():
    assert _parse_llm_entities("no json here") == ([], [])


# --- Guidance policies -----------------------------------------------------------------
@pytest.fixture
def sub():
    return SubgraphResult(paper_ids=["P1", "P2"], method="fuzzy", cypher_used="04_papers_and_authors_by_topic")


PAPER_OF = {"c1": "P1", "c2": "X", "c3": "P2", "c4": "Y"}.get


def test_guidance_filter_keeps_only_subgraph(sub):
    assert apply_guidance(["c1", "c2", "c3", "c4"], sub, paper_of=PAPER_OF, policy="filter") == ["c1", "c3"]


def test_guidance_filter_falls_back_when_empty(sub):
    # No candidate is in the subgraph → return the original ranking, never an empty pool.
    out = apply_guidance(["c2", "c4"], sub, paper_of=PAPER_OF, policy="filter")
    assert out == ["c2", "c4"]


def test_guidance_booster_promotes_subgraph(sub):
    # c3 is in-subgraph but ranked last → boosting lifts it above out-of-subgraph c2/c4.
    out = apply_guidance(["c2", "c4", "c3"], sub, paper_of=PAPER_OF, policy="booster", boost=1.0)
    assert out[0] == "c3"


def test_guidance_expansion_appends_missing_chunks(sub):
    chunks_by_paper = {"P1": ["c1", "cNEW"], "P2": ["c3"]}
    out = apply_guidance(["c1", "c2"], sub, paper_of=PAPER_OF, policy="expansion",
                         chunks_by_paper=chunks_by_paper, max_expand=5)
    assert out[:2] == ["c1", "c2"]      # original ranking preserved
    assert "cNEW" in out and "c3" in out  # pulled missing subgraph chunks in


def test_guidance_noop_on_empty_subgraph():
    empty = SubgraphResult()  # fallback
    assert apply_guidance(["c1", "c2"], empty, paper_of=PAPER_OF, policy="filter") == ["c1", "c2"]


def test_guidance_unknown_policy(sub):
    with pytest.raises(ValueError):
        apply_guidance(["c1"], sub, paper_of=PAPER_OF, policy="bogus")


# ======================================================================================
# Coverage gaps surfaced by the adversarial review (link_spacy, exception paths, route
# fall-through, cache identity, LLM topic re-match, env override, max_expand, parse edges).
# ======================================================================================

# --- GraphIndex cache (identity + eviction) --------------------------------------------
def test_get_index_caches_per_driver(driver):
    a = get_index(driver)
    b = get_index(driver)
    assert a is b                                   # same driver → same cached object
    clear_index_cache()
    assert get_index(driver) is not a               # cleared → rebuilt


# --- select_subgraph exception paths → graceful degradation ----------------------------
class _VocabRaisingDriver:
    """driver.session() works syntactically but every query raises → get_index fails."""

    class _S:
        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def run(self, *a, **k):
            raise RuntimeError("graph down")

    def session(self):
        return self._S()


class _RouteRaisingDriver(_FakeDriver):
    """Answers vocabulary queries (so get_index works) but raises on pid/routing queries."""

    class _S(_FakeSession):
        def run(self, cypher, **params):
            if params.get("names") or params.get("codes"):
                raise RuntimeError("routing query failed")
            return super().run(cypher, **params)

    def session(self):
        return self._S(self.graph)


def test_select_subgraph_get_index_failure_is_fallback():
    clear_index_cache()
    res = select_subgraph("papers by Ada Lovelace", neo4j_driver=_VocabRaisingDriver(), method="fuzzy")
    assert res.method == "fallback" and res.paper_ids == []


def test_select_subgraph_route_exception_keeps_linker_method():
    clear_index_cache()
    res = select_subgraph("papers by Ada Lovelace", neo4j_driver=_RouteRaisingDriver(), method="fuzzy")
    # linking succeeded → method stays "fuzzy" (not "fallback"); no usable subgraph → empty.
    assert res.method == "fuzzy" and res.paper_ids == []
    assert "Ada Lovelace" in res.seed_nodes


def test_select_subgraph_auto_respects_env(driver, monkeypatch):
    monkeypatch.setenv("CSAI415_GRAPH_LINK", "spacy")
    # spaCy may be unavailable in CI; if so it must still not crash (degrades to fallback).
    res = select_subgraph("What did Ada Lovelace write?", neo4j_driver=driver, method="auto")
    assert res.method in {"spacy", "fallback"}


# --- route_subgraph fall-through when a template fires but returns no papers ------------
class _CollabEmptyDriver(_FakeDriver):
    """Collaboration template returns empty → router must fall through to author-only (01)."""

    class _S(_FakeSession):
        def _route(self, cypher, params):
            if "co:Author" in cypher:
                return []  # collab yields nothing
            return super()._route(cypher, params)

    def session(self):
        return self._S(self.graph)


def test_route_collab_empty_falls_through_to_author(index):
    clear_index_cache()
    drv = _CollabEmptyDriver()
    q = "Who does Ada Lovelace collaborate with?"
    pids, cypher, _ = route_subgraph(link_fuzzy(q, index), drv, q)
    assert cypher == "01_papers_by_author"          # fell through from empty collab
    assert set(pids) == {"pA1", "pA2"}


# --- spaCy linker (real model if installed, else skip) ---------------------------------
def test_link_spacy_links_person_to_author(index):
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    hits = link_spacy("What has Ada Lovelace contributed?", index)
    assert any(h.label == "Author" and h.name == "Ada Lovelace" for h in hits)


# --- LLM linker topic re-matching (mocked backend) -------------------------------------
def _fake_client(content):
    msg = type("M", (), {"content": content})
    choice = type("C", (), {"message": msg})
    resp = type("R", (), {"choices": [choice]})
    completions = type("Cmp", (), {"create": lambda self, **kw: resp})()
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


def test_link_llm_rematches_topic_through_synonym_table(index, monkeypatch):
    monkeypatch.setattr(
        "csai415.answer.resolve_backend",
        lambda: (_fake_client('{"authors": ["Ada Lovelace"], "topics": ["natural language processing"]}'),
                 "fake-model", "ollama"))
    hits = link_llm("anything", index)
    names = {(h.label, h.name) for h in hits}
    assert ("Author", "Ada Lovelace") in names
    assert ("Topic", "cs.CL") in names               # "natural language processing" → cs.CL


# --- fuzzy threshold boundary ----------------------------------------------------------
def test_fuzzy_threshold_gates_match(index):
    typo = ["Ada Lovlace"]                            # ~one-edit from "Ada Lovelace"
    assert _fuzzy_link_authors(typo, index, threshold=80)        # lenient → links
    assert _fuzzy_link_authors(typo, index, threshold=100) == [] # exact-only → rejects


# --- apply_guidance: booster with explicit scores, expansion cap & empty inputs --------
def test_guidance_booster_with_explicit_scores(sub):
    # c1(P1, in-sub) vs c2(X, out). Out-of-sub has higher base score, but boost flips it.
    scores = {"c1": 0.1, "c2": 0.9}
    out = apply_guidance(["c2", "c1"], sub, paper_of=PAPER_OF, policy="booster",
                         scores=scores, boost=1.0)
    assert out[0] == "c1"


def test_guidance_expansion_respects_max_expand(sub):
    chunks_by_paper = {"P1": ["c1", "x1", "x2", "x3"], "P2": ["x4", "x5"]}
    out = apply_guidance(["c1"], sub, paper_of=PAPER_OF, policy="expansion",
                         chunks_by_paper=chunks_by_paper, max_expand=2)
    assert len(out) == 1 + 2                          # original + exactly max_expand injected


def test_guidance_expansion_empty_chunks_by_paper_is_noop(sub):
    assert apply_guidance(["c1", "c2"], sub, paper_of=PAPER_OF, policy="expansion",
                          chunks_by_paper={}) == ["c1", "c2"]
    assert apply_guidance(["c1", "c2"], sub, paper_of=PAPER_OF, policy="expansion",
                          chunks_by_paper=None) == ["c1", "c2"]


# --- _parse_llm_entities edge cases ----------------------------------------------------
def test_parse_llm_entities_missing_topics_key():
    assert _parse_llm_entities('{"authors": ["Ada"]}') == (["Ada"], [])


def test_parse_llm_entities_coerces_and_filters():
    a, t = _parse_llm_entities('{"authors": [" Ada ", "", 123], "topics": null}')
    assert a == ["Ada", "123"] and t == []           # stripped, empties dropped, ints coerced
