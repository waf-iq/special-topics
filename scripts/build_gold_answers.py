"""Build gold eval set (qa_answers.jsonl) and QLoRA train set (qa_train.jsonl).

Owner: Ahmad Fraij (Task 4, D3/D4)

Generation method: Groq llama-3.3-70b-versatile prompted to produce grounded
Q/A pairs from arXiv cs.CL chunk texts. All pairs manually verifiable against
the source chunks in data/processed/chunks.parquet.

Gold set  (data/gold/qa_answers.jsonl):  40 rows · 20 papers · 2 Q/A each
Train set (data/train/qa_train.jsonl):  125 rows · 25 papers · 5 Q/A each
Leakage:  zero question overlap between sets (asserted at eval time in eval.py)

Usage:
    export GROQ_API_KEY=<key>
    python scripts/build_gold_answers.py [--gold-only] [--train-only]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
from groq import Groq

GOLD_PAPERS = [
    "2605.30465v1", "2605.30478v1", "2605.30487v1", "2605.30497v1", "2605.30501v1",
    "2605.30504v1", "2605.30521v1", "2605.30526v1", "2605.30568v1", "2605.30574v1",
    "2605.30580v1", "2605.30640v1", "2605.30653v1", "2605.30711v1", "2605.30727v1",
    "2605.30790v1", "2605.31010v1", "2605.31042v1", "2605.31105v1", "2605.31183v1",
]
EXTRA_TRAIN_PAPERS = [
    "2605.30481v1", "2605.30514v1", "2605.30523v1", "2605.30545v1", "2605.30599v1",
]

GOLD_SYSTEM = (
    "You are a scientific QA pair generator. Given a chunk of text from an arXiv paper, "
    "generate exactly 2 question-answer pairs that:\n"
    "1. Are answerable ONLY from the given chunk text (no outside knowledge needed)\n"
    "2. Are specific and factual (not vague or general)\n"
    "3. Have a concise reference answer (1-3 sentences, grounded in the text)\n\n"
    'Output ONLY valid JSON array with exactly 2 objects, each with keys: "question", "answer"\n'
    "No markdown, no explanation."
)

TRAIN_SYSTEM = (
    "You are a scientific QA training data generator for fine-tuning a language model. "
    "Given a chunk of text from an arXiv paper and 2-3 additional context chunks, generate "
    "5 diverse training examples. Each example must:\n"
    "1. Have a question answerable from the provided chunk text\n"
    "2. Have a grounded answer that cites its sources with [1] [2] notation referencing the chunks\n"
    "3. Cover different aspects (methods, results, definitions, comparisons)\n\n"
    'Output ONLY valid JSON array with exactly 5 objects, each with keys: "question", "answer"\n'
    "No markdown, no explanation."
)


def _call_groq(client: Groq, system: str, prompt: str, max_tokens: int = 900) -> list[dict]:
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as exc:
            if attempt == 3:
                raise
            delay = 6 * (2 ** attempt)
            print(f"  retry {attempt + 1} ({exc}) — sleeping {delay}s")
            time.sleep(delay)
    return []


def build_gold(client: Groq, arxiv: pd.DataFrame, out_path: str) -> list[dict]:
    gold = []
    qid = 1
    for pid in GOLD_PAPERS:
        chunks = arxiv[arxiv["paper_id"] == pid].sort_values("chunk_id")
        title = chunks["title"].iloc[0]
        mid = len(chunks) // 3
        row = chunks.iloc[mid]
        chunk_text = row["text"][:1200]
        ps = int(row["page_start"]) if pd.notna(row["page_start"]) else 1
        pe = int(row["page_end"]) if pd.notna(row["page_end"]) else ps
        page_range = str(ps) if ps == pe else f"{ps}-{pe}"

        pairs = _call_groq(client, GOLD_SYSTEM, f"Paper: {title}\n\nChunk text:\n{chunk_text}", 600)
        for pair in pairs[:2]:
            gold.append({
                "qid": f"a{qid:03d}",
                "question": pair["question"],
                "reference_answer": pair["answer"],
                "relevant_chunk_ids": [row["chunk_id"]],
                "paper_id": pid,
                "page_range": page_range,
                "source": "arxiv",
                "source_chunk_text": chunk_text[:300],
            })
            qid += 1
        print(f"  gold [{qid-2}-{qid-1}] {pid}: {title[:55]}")
        time.sleep(2.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for row in gold:
            f.write(json.dumps(row) + "\n")
    print(f"Written {len(gold)} rows → {out_path}")
    return gold


def build_train(client: Groq, arxiv: pd.DataFrame, gold: list[dict], out_path: str) -> None:
    gold_paper_chunks: dict[str, set] = {}
    for r in gold:
        gold_paper_chunks.setdefault(r["paper_id"], set()).add(r["relevant_chunk_ids"][0])
    seen_q = {r["question"].lower().strip() for r in gold}

    train = []
    qid = 1

    all_papers = [(pid, gold_paper_chunks.get(pid, set())) for pid in GOLD_PAPERS] + \
                 [(pid, set()) for pid in EXTRA_TRAIN_PAPERS]

    for pid, used_chunks in all_papers:
        chunks = arxiv[arxiv["paper_id"] == pid].sort_values("chunk_id")
        title = chunks["title"].iloc[0]
        available = chunks[~chunks["chunk_id"].isin(used_chunks)]
        if len(available) < 2:
            available = chunks
        start = len(available) // 2
        sample = available.iloc[start: start + 3]
        contexts = [r["text"][:600] for _, r in sample.iterrows()]
        prompt = f"Paper: {title}\n\nChunk [1]:\n{contexts[0]}"
        for i, ctx in enumerate(contexts[1:], 2):
            prompt += f"\n\nChunk [{i}]:\n{ctx}"

        pairs = _call_groq(client, TRAIN_SYSTEM, prompt, 900)
        added = 0
        for pair in pairs[:5]:
            q = pair["question"].lower().strip()
            if q in seen_q:
                continue
            train.append({
                "qid": f"t{qid:03d}",
                "question": pair["question"],
                "contexts": contexts,
                "answer": pair["answer"],
                "paper_ids": [pid],
                "source": "arxiv",
            })
            seen_q.add(q)
            qid += 1
            added += 1
        print(f"  train +{added} [{qid-added}-{qid-1}] {pid}: {title[:50]}")
        time.sleep(2.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    print(f"Written {len(train)} rows → {out_path}")

    # Leakage assertion
    gold_q = {r["question"].lower().strip() for r in gold}
    train_q = {r["question"].lower().strip() for r in train}
    overlap = gold_q & train_q
    assert not overlap, f"Leakage! {len(overlap)} shared questions"
    print("Leakage check PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--chunks", default="data/processed/chunks.parquet")
    parser.add_argument("--gold-out", default="data/gold/qa_answers.jsonl")
    parser.add_argument("--train-out", default="data/train/qa_train.jsonl")
    args = parser.parse_args()

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    df = pd.read_parquet(args.chunks)
    arxiv = df[df["source"] == "arxiv"].copy()

    if not args.train_only:
        gold = build_gold(client, arxiv, args.gold_out)
    else:
        with open(args.gold_out) as f:
            gold = [json.loads(l) for l in f]

    if not args.gold_only:
        build_train(client, arxiv, gold, args.train_out)


if __name__ == "__main__":
    main()
