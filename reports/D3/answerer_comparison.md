# D3 · Task 3 — Answerer Strategy Comparison

**Owner:** WAFIQ · **Deliverable:** D3 Phase A (zero-shot answerer) · **Source:** `notebooks/04_answerer.ipynb` (`code-compare`)

**Setup.** 7 grounded dev questions (1 relevant context + 2 distractors each) × 2 backends ×
3 citation strategies. `faith_proxy` = lexical overlap of the answer with its cited contexts
(a stand-in until RAGAS at integration). Latency measured **warm** on a Colab **T4** GPU (a one-time
~60 s model load is excluded via a warmup call). Temperature 0.

| Backend | Role | Strategy | Faithfulness (proxy) | Avg. citations | Median latency (ms) | p95 latency (ms) |
|:--|:--|:--|--:|--:|--:|--:|
| Groq Llama-3.3-70B | ceiling | numbered | 0.746 | 1.00 | 295.5 | 408.4 |
| Groq Llama-3.3-70B | ceiling | hybrid   | 0.762 | 1.00 | 322.7 | 610.0 |
| Groq Llama-3.3-70B | ceiling | posthoc  | 0.746 | 1.00 | 412.2 | 528.4 |
| **Qwen2.5-3B-Instruct** | **graded** | **numbered** | **0.916** | **1.00** | **957.7** | **1222.2** |
| **Qwen2.5-3B-Instruct** | **graded** | **hybrid**   | **0.916** | **1.00** | **849.4** | **954.7** |
| **Qwen2.5-3B-Instruct** | **graded** | **posthoc**  | **0.916** | **1.00** | **859.1** | **1203.9** |

## How to read it

- **Citation strategy is a no-op on clean retrieval.** `numbered ≈ hybrid ≈ posthoc` for both models —
  because neither miscites, so the cite→verify→repair verifier has nothing to drop. The verifier is
  *correct, just idle*: on a deliberately miscited input it removes the bad `[2]` (`drop`) or corrects
  it to `[1]` (`reattribute`) — see `notebooks/04_answerer.ipynb` §2b.
- **Winner: `numbered`** — simplest and lowest-latency, and as faithful as the alternatives here.
- **Qwen-3B meets the p95 ≤ 2 s target** (~0.86–0.96 s median warm). Groq is faster but it is the free
  quality-ceiling row, **not** the graded answerer.
- **`avg_citations = 1.00` for Qwen** after two cheap fixes: prompt wording (concrete `[1]` example,
  forbid the literal `n`) + lenient citation parsing (recover malformed markers like `[B1]`, `[n.1]`).
  The residual generation drift motivates the **D4 QLoRA fine-tune**.
- **`faith_proxy` caveat:** Qwen scores *higher* than Groq only because it answers more **extractively**
  (copies context wording → more lexical overlap) while Groq paraphrases. The proxy rewards copying, not
  faithfulness — to be replaced by the **RAGAS** semantic judge at integration.
