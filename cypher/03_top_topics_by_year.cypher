// Intent: Find the top 10 topics by paper count within a given year window.
// Usage: Replace 2020 and 2024 with your desired range.
// Expected output shape: (topic_name, paper_count) — top 10.
// Why useful for D3 GraphRAG agent: Supports trend queries like
// "What topics were most active between 2021 and 2023?"
// Gives the agent a high-level map of the corpus before drilling down.

MATCH (p:Paper)-[:ABOUT]->(t:Topic)
WHERE p.year >= 2020 AND p.year <= 2024
RETURN t.name           AS topic,
       COUNT(DISTINCT p) AS paper_count
ORDER BY paper_count DESC
LIMIT 10;