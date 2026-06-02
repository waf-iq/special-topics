# D2 Cypher Query Examples

Captured post D2-INT1. Graph: 155 Paper nodes, 764 Author nodes, 13 Topic nodes.

---

## Query 01 — Papers by a given author

*File: `cypher/01_papers_by_author.cypher`*

```cypher
MATCH (a:Author {name: "Jun Wang"})-[:WROTE]->(p:Paper)
            RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year
            ORDER BY p.year DESC
```

| paper_id | title | year |
| --- | --- | --- |
| 2605.30947v1 | Extending AI for Research to the Humanities: A Multi-Agent Framework for Evidence-Grounded Scholarship | 2026 |
| 2605.31196v1 | Probing Collision Grounding in Vision-Language Models for Safe Human-Robot Collaboration | 2026 |

_(showing up to 10 rows)_

---

## Query 02 — Top co-authors

*File: `cypher/02_top_coauthors.cypher`*

```cypher
MATCH (a:Author {name: "Jun Wang"})-[:WROTE]->(p:Paper)<-[:WROTE]-(coauthor:Author)
            WHERE coauthor.name <> "Jun Wang"
            RETURN coauthor.name AS coauthor, COUNT(DISTINCT p) AS shared_papers
            ORDER BY shared_papers DESC LIMIT 10
```

| coauthor | shared_papers |
| --- | --- |
| Yating Pan | 1 |
| Qi Su | 1 |
| Jiajun Zhang | 1 |
| Xiaohao Xu | 1 |
| Xiaonan Huang | 1 |

_(showing up to 10 rows)_

---

## Query 03 — Top topics by year window (2020–2024)

*File: `cypher/03_top_topics_by_year.cypher`*

```cypher
MATCH (p:Paper)-[:ABOUT]->(t:Topic)
            WHERE p.year >= 2025 AND p.year <= 2026
            RETURN t.name AS topic, COUNT(DISTINCT p) AS paper_count
            ORDER BY paper_count DESC LIMIT 10
```

| topic | paper_count |
| --- | --- |
| cs.CL | 113 |
| cs.LG | 10 |
| cs.CV | 10 |
| cs.CR | 5 |
| cs.AI | 4 |
| cs.SE | 3 |
| cs.IR | 3 |
| cs.CY | 2 |
| cs.GT | 1 |
| cs.MM | 1 |

_(showing up to 10 rows)_

---

## Query 04 — Papers and authors by topic (cs.CL)

*File: `cypher/04_papers_and_authors_by_topic.cypher`*

```cypher
MATCH (a:Author)-[:WROTE]->(p:Paper)-[:ABOUT]->(t:Topic {name: "cs.CL"})
            RETURN p.title AS paper_title, p.year AS year, a.name AS author
            ORDER BY p.year DESC, p.title LIMIT 10
```

| paper_title | year | author |
| --- | --- | --- |
| "Intelegi Româneşte?'' A Recipe for Romanian Vision-Language Models | 2026 | Traian Rebedea |
| "Intelegi Româneşte?'' A Recipe for Romanian Vision-Language Models | 2026 | Mihai Dascalu |
| "Intelegi Româneşte?'' A Recipe for Romanian Vision-Language Models | 2026 | Mihai Masala |
| "Intelegi Româneşte?'' A Recipe for Romanian Vision-Language Models | 2026 | Marius Leordeanu |
| A Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation | 2026 | Zhe Hu |
| A Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation | 2026 | Yushi Li |
| A Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation | 2026 | Jing Li |
| A Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation | 2026 | Siqi Wang |
| A Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation | 2026 | Yi Zhao |
| AI for Monitoring and Classifying Data Used in Research Literature | 2026 | Rafael Macalaba |

_(showing up to 10 rows)_

---

## Query 05 — Authors on both cs.CL and cs.LG

*File: `cypher/05_authors_on_both_topics.cypher`*

```cypher
MATCH (a:Author)-[:WROTE]->(p1:Paper)-[:ABOUT]->(t1:Topic {name: "cs.CL"})
            MATCH (a)-[:WROTE]->(p2:Paper)-[:ABOUT]->(t2:Topic {name: "cs.CV"})
            RETURN a.name AS author, COUNT(DISTINCT p1) AS cl_papers,
                   COUNT(DISTINCT p2) AS lg_papers
            ORDER BY (cl_papers + lg_papers) DESC LIMIT 20
```

| author | cl_papers | lg_papers |
| --- | --- | --- |
| Jun Wang | 1 | 1 |

_(showing up to 10 rows)_
