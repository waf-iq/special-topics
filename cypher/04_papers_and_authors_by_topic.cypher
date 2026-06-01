// Intent: Find all papers about a given topic and the authors who wrote them.
// Usage: Replace "cs.CL" with any topic name.
// Expected output shape: (paper_title, year, author_name) rows.
// Why useful for D3 GraphRAG agent: Answers "Who works on topic T?"
// The agent can use this to surface relevant experts when a user
// asks about a research area.

MATCH (a:Author)-[:WROTE]->(p:Paper)-[:ABOUT]->(t:Topic {name: "cs.CL"})
RETURN p.title   AS paper_title,
       p.year    AS year,
       a.name    AS author
ORDER BY p.year DESC, p.title;