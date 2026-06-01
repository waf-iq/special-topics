# D2 Graph Schema (Neo4j)

Knowledge graph built by `scripts/seed_neo4j.py` (D2-C1). Source of truth is the
Mongo `papers` collection (D2-A2); D3's GraphRAG agent queries this graph via the
Cypher library in `cypher/` (D2-C2).

## Nodes

| Label | Properties | Key (unique constraint) | Notes |
|---|---|---|---|
| `Paper` | `paper_id`, `title`, `year`, `source` | `paper_id` | `source ∈ {scifact, arxiv}`; `year` may be `null`. |
| `Author` | `name` | `name` | Dedup is **name-based** — two distinct people sharing a name collapse into one node (known limitation, see Pitfalls). |
| `Topic` | `name` | `name` | arXiv category code, e.g. `cs.CL`, `cs.AI`. From `papers.topics` (full category list). |

`Venue` is in the H0 contract but not emitted by this loader — arXiv carries no
venue metadata for the current corpus. Added trivially if venue data appears.

## Relationships

| Edge | Direction | Meaning |
|---|---|---|
| `WROTE` | `(:Author)-[:WROTE]->(:Paper)` | Author is a listed author of the paper. |
| `ABOUT` | `(:Paper)-[:ABOUT]->(:Topic)` | Paper is categorized under the topic. |

No `CITES` edges — the arXiv API doesn't expose citation data. Deferred to D3
(brief permits "add CITES if time").

## Constraints (applied at load, idempotent)

```cypher
CREATE CONSTRAINT paper_id    IF NOT EXISTS FOR (p:Paper)  REQUIRE p.paper_id IS UNIQUE;
CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name     IS UNIQUE;
CREATE CONSTRAINT topic_name  IF NOT EXISTS FOR (t:Topic)  REQUIRE t.name      IS UNIQUE;
```

## Loading semantics

- **Idempotent:** every write is `MERGE`, never `CREATE`. Re-running the seed is
  safe and produces no duplicates.
- **SciFact skipped:** SciFact papers have no author metadata (BEIR ships none),
  so they have empty `authors` and are skipped — loading them would create
  authorless `Paper` nodes with no `WROTE` edges. The skipped count is logged.
- **Source enum:** parquet's `arxiv-demo` / `arxiv` are both stored as `arxiv`
  on `Paper.source`, matching the Mongo contract enum.

## Running the loader

```bash
# Primary path — reads Mongo `papers` (requires D2-A2's seed + a running Neo4j):
python scripts/seed_neo4j.py --source all

# Dev fallback — reads chunks.parquet directly, before D2-A2's Mongo seed lands:
python scripts/seed_neo4j.py --from-parquet data/processed/chunks.parquet --source arxiv

# Inspect without writing:
python scripts/seed_neo4j.py --from-parquet data/processed/chunks.parquet --dry-run
```

Connection config comes from env (`MONGO_URL`, `NEO4J_URL`, `NEO4J_USER`,
`NEO4J_PASSWORD`) or matching CLI flags. Dev containers run `NEO4J_AUTH=none`,
so leave the password unset.

## Current corpus stats (dev parquet fallback, pre-expansion)

5 arxiv-demo + 2 arxiv papers → **7 papers, 38 authors, 2 topics** (`cs.CL`,
`cs.AI`), 38 `WROTE` edges, 7 `ABOUT` edges. After D2-A1's ~150-paper expansion
and the D2-INT1 reseed these grow to ~120 papers / ~300+ authors.

## Example graph shape

```
(:Author {name:"Ziyu Guo"})-[:WROTE]->(:Paper {paper_id:"2605.15198v1",
                                                title:"...", year:2026,
                                                source:"arxiv"})-[:ABOUT]->(:Topic {name:"cs.CL"})
```

## Pitfalls

- **Name-based author dedup:** homonymous authors collapse into one node. A
  proper fix needs ORCID or affiliation disambiguation — out of scope for D2.
- **Single-topic skew:** if the corpus is filtered to `cat:cs.CL`, almost every
  paper hangs off one `Topic` node. D2-C2 query 3 (year-window counts) stays
  non-trivial in that case.
