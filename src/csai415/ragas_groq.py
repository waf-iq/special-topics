"""RAGAS faithfulness + answer-relevance judged by Groq llama-3.3-70b.

Drop-in for evaluate_answers() in eval.py. Uses langchain-groq as the LLM
backend for RAGAS >=0.2, with exponential backoff for the 30 rpm free tier.
"""

from __future__ import annotations

import os
import time


def _llm_and_embeddings():
    """Return (llm, embeddings) for RAGAS, backed by Groq + HuggingFace."""
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    api_key = os.environ.get("GROQ_API_KEY", "")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0,
        max_retries=6,
    )
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return llm, embeddings


def _with_backoff(fn, retries: int = 6, base_delay: float = 4.0):
    """Call fn() with exponential backoff on RateLimitError / 429."""
    import groq

    for attempt in range(retries):
        try:
            return fn()
        except (groq.RateLimitError, Exception) as exc:
            if attempt == retries - 1:
                raise
            is_rate = "429" in str(exc) or "rate" in str(exc).lower()
            delay = base_delay * (2**attempt) if is_rate else base_delay
            time.sleep(delay)


def ragas_score(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict[str, float]:
    """Run RAGAS faithfulness + answer_relevancy via Groq.

    Args:
        questions: list of query strings
        answers:   list of model-generated answer strings
        contexts:  list of context lists (one list of chunk texts per question)
        ground_truths: optional reference answers (improves answer_relevancy)

    Returns:
        {"faithfulness": float, "answer_relevancy": float}
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    llm, embeddings = _llm_and_embeddings()

    data: dict[str, list] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    def _run():
        return evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

    result = _with_backoff(_run)
    df = result.to_pandas()

    faith_val = float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0
    relev_val = float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else 0.0
    return {"faithfulness": faith_val, "answer_relevancy": relev_val}
