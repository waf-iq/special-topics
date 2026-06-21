# Task 5 - GraphRAG Ablation + Multi-objective HPO (D3)

**Owner:** Yehia Noureldin  
**Runner:** `src/csai415/graphrag_hpo.py`  
**Answerer:** `qwen2.5:3b-instruct` (zero-shot)  
**Judge:** RAGAS + Groq `llama-3.3-70b-versatile`

## 1. Ablation grid (3 modes x rerank)

| mode | rerank | faithfulness | answer_relevance | recall@5 | p95_latency_ms | n |
|---|---|---|---|---|---|---|
| vector | False | 0.966 | 0.905 | 0.400 | 5106 | 10 |
| vector | True | 0.750 | 0.874 | 0.600 | 18105 | 10 |
| graph | False | nan | 0.936 | 0.500 | 4221 | 10 |
| graph | True | nan | nan | 0.600 | 13912 | 10 |
| hybrid | False | nan | nan | 0.500 | 4314 | 10 |
| hybrid | True | nan | nan | 0.600 | 12961 | 10 |

**Winning mode:** `vector` - highest faithfulness, latency tie-break.

## 2. NSGA-II Pareto front (faithfulness up / latency down)

| trial | rerank | candidate_k | rerank_top_n | k | faithfulness | p95_latency_ms | knee? |
|---|---|---|---|---|---|---|---|

![Pareto front](pareto_front.png)

## 3. Recommended config (knee point)

```
{'mode': 'vector'}
```

## 4. Task 4 headline (production row)
faithfulness **0.966** / answer_relevance **0.905** - targets >=0.8/>=0.8: **MET**.
