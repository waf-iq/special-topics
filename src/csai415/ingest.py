"""Pair A — Ingest pipeline. See MEMBER_BRIEF.md §6.A.

D1 corpus = SciFact (BEIR) via ir_datasets (~5,183 abstracts, 300 manually-judged
test claims) PLUS 5 arXiv cs.CL PDFs for an end-to-end PDF-pipeline demo. The
arXiv chunks are marked source="arxiv-demo" and are not used in evaluation.

Outputs:
  data/processed/chunks.parquet  (schema in MEMBER_BRIEF.md §5)
  data/gold/qa.jsonl             (SciFact test claims + qrels mapped to chunk_ids)
"""

from __future__ import annotations

from pathlib import Path

DATA_RAW = Path("data/raw_pdfs")
DATA_PROCESSED = Path("data/processed")
DATA_GOLD = Path("data/gold")

CHUNKS_PARQUET = DATA_PROCESSED / "chunks.parquet"
GOLD_JSONL = DATA_GOLD / "qa.jsonl"

ARXIV_DEMO_COUNT = 5
SCIFACT_SPLIT = "test"  # 300 evaluated claims live in the test split


def load_scifact() -> "pd.DataFrame":
    """Load SciFact (BEIR) via ir_datasets and return a DataFrame ready for chunking.

    Each row should carry: paper_id (doc_id from SciFact), text (title + abstract or
    chunks of it), title, plus source='scifact'. Page columns null. See §6.A Q1–Q2.
    """
    raise NotImplementedError(
        "Pair A — use `ir_datasets.load('beir/scifact')`; decide test-only vs train+test."
    )


def download_arxiv_demo(n: int = ARXIV_DEMO_COUNT, query: str = "cat:cs.CL") -> list[Path]:
    """Download a small number of arXiv PDFs into DATA_RAW for the pipeline demo.

    These are NOT used in evaluation — they exist only to prove the PDF parser runs
    end-to-end so D2 doesn't start from zero.
    """
    raise NotImplementedError(
        "Pair A — `arxiv` Python package is the easiest. Sort by submitted_date desc."
    )


def parse_arxiv_pdfs(pdf_paths: list[Path], chunk_tokens: int = 300, overlap: int = 50) -> "pd.DataFrame":
    """Parse and chunk arXiv PDFs with PyMuPDF, preserving page_start/page_end. source='arxiv-demo'."""
    raise NotImplementedError("Pair A — PyMuPDF (fitz) is the speed/fidelity winner. See §6.A Q3.")


def embed_chunks(df: "pd.DataFrame", model_name: str = "sentence-transformers/bge-small-en") -> "pd.DataFrame":
    """Add 'embedding' column (list[float], 384-dim) using the named SBERT model. Batch=32."""
    raise NotImplementedError("Pair A — batch with batch_size=32 to fit CPU.")


def build_gold_from_scifact() -> None:
    """Convert SciFact test-split claims+qrels into qa.jsonl.

    Line format: {"qid": claim_id, "question": claim_text, "relevant_chunk_ids": [...], "topic": null}
    If you sub-chunked abstracts, decide whether ALL chunks of the relevant doc count, or only the
    chunk containing the evidence sentence. See §6.A Q5 — document your choice in the report.
    """
    raise NotImplementedError("Pair A — see SciFact's qrels structure (claim_id → doc_id with label 1/2).")


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_GOLD.mkdir(parents=True, exist_ok=True)

    # Main corpus: SciFact
    scifact_df = load_scifact()

    # Pipeline demo: 5 arXiv PDFs
    pdf_paths = download_arxiv_demo()
    arxiv_df = parse_arxiv_pdfs(pdf_paths)

    # Combine, embed, persist
    import pandas as pd
    df = pd.concat([scifact_df, arxiv_df], ignore_index=True)
    df = embed_chunks(df)
    df.to_parquet(CHUNKS_PARQUET)

    # Gold set comes only from SciFact (real human judgments)
    build_gold_from_scifact()


if __name__ == "__main__":
    main()
