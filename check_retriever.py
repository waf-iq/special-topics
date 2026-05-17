"""Quick sanity check for HybridRetriever. Run from special-topics root: python check_retriever.py"""

from csai415.retrieve import HybridRetriever, RetrieverConfig, load_chunks

df = load_chunks()
print(f"Loaded {len(df)} chunks")

r = HybridRetriever(df, RetrieverConfig())
print("Built indexes (BM25 + dense)")

results = r.search("vitamin D supplementation", k=5, hybrid_weight=0.5)
print("Top 5 chunk_ids:")
for cid in results:
    print(f"  {cid}")
