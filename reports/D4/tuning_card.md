# Tuning Card — `qwen2.5-3b-csai415` (QLoRA)

D4 Task 3b · Owner: WAFIQ · produced by `notebooks/06_qlora_tune.ipynb`.
Fill the `TODO` cells from the notebook's printed output, then commit alongside the run.

## Model
| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Base license | Apache-2.0 |
| Method | QLoRA (4-bit NF4 + double-quant, bf16 compute) |
| Served as | `qwen2.5-3b-csai415` (Ollama, GGUF `Q4_K_M`) |
| Selected by | `CSAI415_ANSWERER=qwen2.5-3b-csai415` (no code change vs zero-shot) |

## Data
| Field | Value |
|---|---|
| Training set | `data/train/qa_train.jsonl` |
| Size | 125 rows (hand-curated arXiv cs.CL Q/A) |
| Leakage check | 0 question overlap with `data/gold/qa_answers.jsonl` ✓ |
| Format | system + numbered-source user template + `[n]`-cited answer — **identical to `src/csai415/answer.py`** |
| Train/eval split | 90/10 (seed 42) |

## Hyperparameters (held constant across the sweep)
| Field | Value |
|---|---|
| LoRA dropout | 0.05 |
| Target modules | q,k,v,o,gate,up,down proj |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine, warmup 0.03) |
| Effective batch | 8 (bs 1 × grad-accum 8) |
| Max seq len | 2048 |
| Optimizer | paged_adamw_8bit |
| Seed | 42 |

## LoRA-rank sweep — compared approaches (TODO — paste Cell 7 table)
Single-axis sweep over LoRA rank at fixed epochs/lr; winner = lowest held-out eval loss.
The per-epoch column is the overfit check on a 125-row set.

| rank | alpha | final eval loss | per-epoch eval losses | train secs |
|---|---|---|---|---|
| 8  | 16 | TODO | TODO | TODO |
| 16 | 32 | TODO | TODO | TODO |
| 32 | 64 | TODO | TODO | TODO |

**Winner:** rank `TODO` — *defend in one line (best eval loss; no overfit turn-up; cost).*

## Run (TODO — fill from notebook)
| Field | Value |
|---|---|
| Hardware (GPU) | TODO |
| Wall-clock | TODO |
| Final train loss | TODO |
| Final eval loss | TODO |
| Adapter size | TODO |
| GGUF size (`Q4_K_M`) | TODO |

## Evaluation (TODO — T7 base-vs-tuned via `evaluate_answers`)
Run the harness twice, flipping `CSAI415_ANSWERER` between `qwen2.5:3b-instruct` (base) and
`qwen2.5-3b-csai415` (tuned). Owned by Abdulrahman (T7); pasted here for the card.

| Metric | Zero-shot (base) | Tuned | Δ |
|---|---|---|---|
| Faithfulness | TODO | TODO | TODO |
| Answer-relevance | TODO | TODO | TODO |
| Recall@5 | TODO | TODO | TODO |
| p95 latency (ms) | TODO | TODO | TODO |

## Discussion (TODO)
- Did the tune improve citation faithfulness without trading away fluency / over-refusing?
- Does 4-bit `Q4_K_M` quantization erode faithfulness vs the merged fp16 model? If so, note the gap and whether `Q5_K_M` is warranted.
- Overfitting watch: eval-loss trajectory across the 3 epochs on a 125-row set.
