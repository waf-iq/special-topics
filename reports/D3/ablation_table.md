# T5 - GraphRAG Ablation Grid (real run)

| mode | rerank | faithfulness | answer_relevance | recall@5 | p95_latency_ms | n |
|---|---|---|---|---|---|---|
| vector | False | 0.966 | 0.905 | 0.400 | 5106 | 10 |
| vector | True | 0.750 | 0.874 | 0.600 | 18105 | 10 |
| graph | False | nan | 0.936 | 0.500 | 4221 | 10 |
| graph | True | nan | nan | 0.600 | 13912 | 10 |
| hybrid | False | nan | nan | 0.500 | 4314 | 10 |
| hybrid | True | nan | nan | 0.600 | 12961 | 10 |
