"""D2-C3 — Run the 5 Cypher queries against the seeded graph and save outputs.

Run after D2-INT1 (graph seeded with ~120 arXiv papers):
    python scripts/capture_cypher_outputs.py

Writes: reports/d2_cypher_examples.md
"""

from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = ""   # blank because docker-compose uses NEO4J_AUTH=none

QUERIES = {
    "01 — Papers by a given author": {
        "file": "cypher/01_papers_by_author.cypher",
        "cypher": """
            MATCH (a:Author {name: "Jun Wang"})-[:WROTE]->(p:Paper)
            RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year
            ORDER BY p.year DESC
        """,
        "columns": ["paper_id", "title", "year"],
    },
    "02 — Top co-authors": {
        "file": "cypher/02_top_coauthors.cypher",
        "cypher": """
            MATCH (a:Author {name: "Jun Wang"})-[:WROTE]->(p:Paper)<-[:WROTE]-(coauthor:Author)
            WHERE coauthor.name <> "Jun Wang"
            RETURN coauthor.name AS coauthor, COUNT(DISTINCT p) AS shared_papers
            ORDER BY shared_papers DESC LIMIT 10
        """,
        "columns": ["coauthor", "shared_papers"],
    },
    "03 — Top topics by year window (2020–2024)": {
        "file": "cypher/03_top_topics_by_year.cypher",
        "cypher": """
            MATCH (p:Paper)-[:ABOUT]->(t:Topic)
            WHERE p.year >= 2025 AND p.year <= 2026
            RETURN t.name AS topic, COUNT(DISTINCT p) AS paper_count
            ORDER BY paper_count DESC LIMIT 10
        """,
        "columns": ["topic", "paper_count"],
    },
    "04 — Papers and authors by topic (cs.CL)": {
        "file": "cypher/04_papers_and_authors_by_topic.cypher",
        "cypher": """
            MATCH (a:Author)-[:WROTE]->(p:Paper)-[:ABOUT]->(t:Topic {name: "cs.CL"})
            RETURN p.title AS paper_title, p.year AS year, a.name AS author
            ORDER BY p.year DESC, p.title LIMIT 10
        """,
        "columns": ["paper_title", "year", "author"],
    },
    "05 — Authors on both cs.CL and cs.LG": {
        "file": "cypher/05_authors_on_both_topics.cypher",
        "cypher": """
            MATCH (a:Author)-[:WROTE]->(p1:Paper)-[:ABOUT]->(t1:Topic {name: "cs.CL"})
            MATCH (a)-[:WROTE]->(p2:Paper)-[:ABOUT]->(t2:Topic {name: "cs.CV"})
            RETURN a.name AS author, COUNT(DISTINCT p1) AS cl_papers,
                   COUNT(DISTINCT p2) AS lg_papers
            ORDER BY (cl_papers + lg_papers) DESC LIMIT 20
        """,
        "columns": ["author", "cl_papers", "lg_papers"],
    },
}


def run_query(session, cypher: str, columns: list[str], limit: int = 10) -> list[dict]:
    result = session.run(cypher)
    rows = []
    for record in result:
        rows.append({col: record.get(col, "") for col in columns})
        if len(rows) >= limit:
            break
    return rows


def rows_to_markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        line = "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |"
        lines.append(line)
    return "\n".join(lines)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    # Get graph stats
    with driver.session() as session:
        paper_count  = session.run("MATCH (p:Paper)  RETURN COUNT(p) AS n").single()["n"]
        author_count = session.run("MATCH (a:Author) RETURN COUNT(a) AS n").single()["n"]
        topic_count  = session.run("MATCH (t:Topic)  RETURN COUNT(t) AS n").single()["n"]

    print(f"Graph stats: {paper_count} Papers, {author_count} Authors, {topic_count} Topics")

    output_lines = [
        "# D2 Cypher Query Examples",
        "",
        f"Captured post D2-INT1. Graph: {paper_count} Paper nodes, "
        f"{author_count} Author nodes, {topic_count} Topic nodes.",
        "",
    ]

    with driver.session() as session:
        for title, q in QUERIES.items():
            print(f"Running: {title}...")
            rows = run_query(session, q["cypher"], q["columns"])

            output_lines += [
                "---",
                "",
                f"## Query {title}",
                "",
                f"*File: `{q['file']}`*",
                "",
                "```cypher",
                q["cypher"].strip(),
                "```",
                "",
                rows_to_markdown_table(rows, q["columns"]) if rows else "_No results._",
                "",
                f"_(showing up to 10 rows)_",
                "",
            ]

    driver.close()

    out_path = Path("reports/d2_cypher_examples.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()