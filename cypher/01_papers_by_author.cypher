// Intent: Find all papers written by a given author.
// Usage: Replace "Yann LeCun" with any author name.
// Expected output shape: list of (paper_id, title, year) rows.
// Why useful for D3 GraphRAG agent: The agent can answer
// "What has author X published?" by running this query and
// returning the paper titles as context.

MATCH (a:Author {name: "Yann LeCun"})-[:WROTE]->(p:Paper)
RETURN p.paper_id   AS paper_id,
       p.title      AS title,
       p.year       AS year
ORDER BY p.year DESC;