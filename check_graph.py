"""Quick sanity check for the live Neo4j graph + select_subgraph.

Run from special-topics root after `docker compose up -d neo4j` and seeding:
    .venv/bin/python check_graph.py

Confirms (1) the driver connects, (2) the seeded graph has the expected
node/edge shape, and (3) select_subgraph(query, neo4j_driver=...) executes
against the live graph without crashing (returns a fallback while the H0 stub
is in place).
"""

import os

from neo4j import GraphDatabase

from csai415.graph_select import select_subgraph

NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")  # dev container runs NEO4J_AUTH=none

auth = (NEO4J_USER, NEO4J_PASSWORD) if NEO4J_PASSWORD else None
driver = GraphDatabase.driver(NEO4J_URL, auth=auth)
driver.verify_connectivity()
print(f"Connected to Neo4j at {NEO4J_URL}")

with driver.session() as s:
    counts = s.run(
        "MATCH (p:Paper) WITH count(p) AS papers "
        "MATCH (a:Author) WITH papers, count(a) AS authors "
        "MATCH (t:Topic) RETURN papers, authors, count(t) AS topics"
    ).single()
    print(f"  Papers={counts['papers']}  Authors={counts['authors']}  Topics={counts['topics']}")

    print("  Sample topics:", [r["name"] for r in s.run(
        "MATCH (t:Topic) RETURN t.name AS name ORDER BY name LIMIT 13")])
    print("  Sample authors:", [r["name"] for r in s.run(
        "MATCH (a:Author)-[:WROTE]->(p) RETURN a.name AS name, count(p) AS n "
        "ORDER BY n DESC LIMIT 5")])

print("\nselect_subgraph against the live driver:")
for q in ["What has cs.CL research shown about attention?", "papers by Yann LeCun"]:
    res = select_subgraph(q, neo4j_driver=driver)
    print(f"  q={q!r}")
    print(f"    method={res.method}  cypher={res.cypher_used}  "
          f"seeds={res.seed_nodes}  n_papers={len(res.paper_ids)}")

driver.close()
print("\nOK — live graph reachable and select_subgraph callable.")
