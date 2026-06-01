// Intent: Find authors who have published on BOTH topic T1 and topic T2.
// Usage: Replace "cs.CL" and "cs.LG" with any two topic names.
// Expected output shape: (author_name, cl_papers, lg_papers).
// Why useful for D3 GraphRAG agent: Identifies cross-disciplinary
// researchers — e.g. someone bridging NLP (cs.CL) and machine learning
// (cs.LG). The agent can use this to answer "Who works at the
// intersection of X and Y?"

MATCH (a:Author)-[:WROTE]->(p1:Paper)-[:ABOUT]->(t1:Topic {name: "cs.CL"})
MATCH (a)-[:WROTE]->(p2:Paper)-[:ABOUT]->(t2:Topic {name: "cs.LG"})
RETURN a.name               AS author,
       COUNT(DISTINCT p1)   AS cl_papers,
       COUNT(DISTINCT p2)   AS lg_papers
ORDER BY (cl_papers + lg_papers) DESC
LIMIT 20;