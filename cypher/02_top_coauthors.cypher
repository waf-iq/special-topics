// Intent: Find the top 10 co-authors of a given author.
// Usage: Replace "Yann LeCun" with any author name.
// Expected output shape: (coauthor_name, shared_paper_count) — top 10.
// Why useful for D3 GraphRAG agent: Supports queries like
// "Who does author X collaborate with most?" — useful for
// recommending related authors or finding research clusters.

MATCH (a:Author {name: "Yann LeCun"})-[:WROTE]->(p:Paper)<-[:WROTE]-(coauthor:Author)
WHERE coauthor.name <> "Yann LeCun"
RETURN coauthor.name        AS coauthor,
       COUNT(DISTINCT p)    AS shared_papers
ORDER BY shared_papers DESC
LIMIT 10;