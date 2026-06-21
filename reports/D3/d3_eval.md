# Task 4 — Data + Evaluation Harness · D3 Report

**Owner:** Ahmad Fraij  
**Files:** `src/csai415/eval.py` · `src/csai415/ragas_groq.py` · `data/gold/qa_answers.jsonl` · `data/train/qa_train.jsonl` · `scripts/build_gold_answers.py`

---

## 1. Gold Eval Set Construction — Method Comparison

Three approaches were considered for building `data/gold/qa_answers.jsonl`.

| Approach | Quality | Speed | Verifiability | Leakage risk | Chosen |
|---|---|---|---|---|---|
| **A — Hand-authored** | Highest — human judges relevance and depth | Slowest (~10 min/paper) | Perfect | None | No |
| **B — Groq-gen + human verify** | High — LLM drafts, human accepts/rewrites | Medium (~2 min/paper) | Good | Low | **Yes (adopted)** |
| **C — Groq-gen only, no review** | Medium — may include trivial or hallucinated Q/A | Fastest (<30 s/paper) | Poor | Medium | No (first pass only) |

**Decision:** Approach B. Groq `llama-3.3-70b-versatile` generated candidate Q/A pairs from selected chunk texts; a manual review pass identified and replaced 10 trivial questions (citation metadata, boilerplate config values) with substantive questions drawn from different chunks in the same paper.

**Why not hand-authored (A)?** With 40 pairs across 20 papers the time cost is acceptable, but the bigger issue is consistency — a human author naturally gravitates toward the most famous result in each paper, biasing the eval set toward headline numbers rather than method understanding. Groq with a strict prompt samples more uniformly across the chunk content.

**Why not Groq-only (C)?** The first generation pass produced questions like "What version of PyTorch is used?" and "In what year was GloVe published?" — trivia that tests lexical recall, not comprehension. A zero-human-review policy would have kept these in, inflating Recall@5 without measuring anything meaningful.

### Gold set statistics

| Metric | Value |
|---|---|
| Total rows | 40 |
| Papers covered | 20 (all arXiv cs.CL) |
| Questions per paper | 2 (from different chunk positions: 1/3 and 2/3 of paper) |
| Avg question length | 12.8 words |
| Avg reference answer length | 16.3 words |
| Multi-page citations | 14 / 40 |
| Trivial questions after review | 0 |

Topics covered span: RAG pipelines, LoRA/PEFT, KV-cache compression, LLM watermarking, alignment, speculative decoding, multi-agent calibration, agent memory, privacy in RAG, LLM-as-a-Judge, safety alignment, knowledge graph retrieval, and code generation with RL.

---

## 2. RAGAS-with-Groq vs Hand-Rolled Judge

The H0 scaffold used a lexical-overlap proxy for faithfulness and answer relevance — cheap but unreliable. Two judging strategies were compared before committing to RAGAS.

| Property | Hand-rolled (lexical overlap) | RAGAS + Groq `llama-3.3-70b` |
|---|---|---|
| **Faithfulness measure** | Word overlap: answer ∩ contexts / \|answer\| | LLM checks each claim in the answer against the context; outputs 0–1 |
| **Answer relevance measure** | Word overlap: answer ∩ question / \|answer\| | LLM embeds question + generates back-questions from answer; cosine similarity |
| **Sensitivity to hallucination** | Low — a fluent off-topic answer scores well if it shares stop-words | High — LLM explicitly marks unsupported claims |
| **Sensitivity to paraphrase** | Low — exact word mismatch penalises correct paraphrase | High — semantic similarity tolerates correct rephrasing |
| **Rate limit / cost** | None | Groq free tier: ~30 rpm / 1 K rpd / 12 K tpm; exponential backoff required |
| **Reproducibility** | Deterministic | Non-deterministic (temperature=0 reduces but doesn't eliminate variance) |
| **Targets (brief)** | — | ≥ 0.8 faithfulness · ≥ 0.8 answer relevance |

**Verdict:** RAGAS + Groq is the only credible option for faithfulness. The overlap proxy systematically over-scores answers that borrow surface words from the context without being grounded in it, and under-scores correct paraphrases. The 30 rpm cap is handled with exponential backoff (base delay 4 s, up to 6 retries) in `ragas_groq.py`.

**Retained overlap code:** the `_overlap` helper was removed from `eval.py` since RAGAS is now the sole judge. For the D4 base-vs-tuned comparison, both models run through the same RAGAS harness, so the metric is directly comparable.

---

## 3. QLoRA Train Set — Design Decisions

`data/train/qa_train.jsonl` is the training corpus for WAFIQ's Phase B QLoRA fine-tune.

| Property | Value |
|---|---|
| Total rows | 125 |
| Papers covered | 25 (20 gold papers + 5 new) |
| Questions per paper | 5 |
| Format | `question` · `contexts` (list of chunk texts) · `answer` (cites [1][2]) · `paper_ids` |
| Leakage vs gold | **0 shared questions** (asserted in `evaluate_answers()` at runtime) |

**Key design choices:**

- **Different chunks from gold.** For the 20 papers shared with the gold set, train Q/A were generated from the second half of the paper — chunks not used for the gold eval. This prevents train/test contamination at the chunk level, not just the question level.
- **Multi-context format.** Each train row provides 2–3 chunk texts as `contexts`. The answer is grounded in these with `[1]` / `[2]` citation notation — the same format `answer.py` produces at inference time, so the model learns the output distribution it will actually need to reproduce.
- **5 extra papers.** To reach 125 rows and add diversity, 5 papers not in the gold set were included (`2605.30481v1`, `2605.30514v1`, `2605.30523v1`, `2605.30545v1`, `2605.30599v1`).

**Trade-off — static vs live-retrieved contexts:**  
The current `contexts` field contains the chunks fed to Groq at generation time, not what the live retriever would return for each question. For maximum training fidelity the contexts should be re-retrieved from the live Qdrant stack (`enrich_train_contexts.py`, deferred to D4). The mismatch is expected to be small since the generation chunks overlap heavily with top-k retrieval results for the same paper.

---

## 4. Leakage Check

The brief requires a hard assertion that train and gold sets are disjoint. This runs automatically inside `evaluate_answers()` before every eval:

```python
def _assert_no_leakage(gold, train_path="data/train/qa_train.jsonl"):
    gold_q = {r["question"].lower().strip() for r in gold}
    train_q = {r["question"].lower().strip() for r in train_rows}
    overlap = gold_q & train_q
    assert not overlap, f"Train/eval leakage — {len(overlap)} shared question(s)"
```

**Result:** 0 shared questions. The check is case-insensitive and strip-normalised to catch near-duplicates.

---

## 5. Summary

| Deliverable | Status |
|---|---|
| Real RAGAS faithfulness/relevance via Groq | ✅ Done — `ragas_groq.py`, wired into `evaluate_answers()` |
| `qa_answers.jsonl` committed (40 rows, 0 trivial) | ✅ Done |
| `qa_train.jsonl` committed (125 rows, disjoint) | ✅ Done |
| Leakage assertion in harness | ✅ Done — fires on every `evaluate_answers()` call |
| Retry/backoff for 30 rpm Groq cap | ✅ Done — exponential backoff, 6 retries |
| Comparison table: gold construction methods | ✅ This document |
| Comparison table: RAGAS vs hand-rolled judge | ✅ This document |
