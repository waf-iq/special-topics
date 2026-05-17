"""Pair A — Ingest pipeline. See MEMBER_BRIEF.md §6.A.

D1 corpus = SciFact (BEIR) via ir_datasets (~5,183 abstracts, 300 manually-judged
test claims) PLUS 5 arXiv cs.CL PDFs for an end-to-end PDF-pipeline demo.
The arXiv chunks are marked source="arxiv-demo" and are not used in evaluation.

Outputs:
  data/processed/chunks.parquet   (schema in MEMBER_BRIEF.md §5)
  data/gold/qa.jsonl              (SciFact test claims + qrels mapped to chunk_ids)
"""

from __future__ import annotations

from pathlib import Path

DATA_RAW = Path("data/raw_pdfs")
DATA_PROCESSED = Path("data/processed")
DATA_GOLD = Path("data/gold")

CHUNKS_PARQUET = DATA_PROCESSED / "chunks.parquet"
GOLD_JSONL = DATA_GOLD / "qa.jsonl"

ARXIV_DEMO_COUNT = 5
SCIFACT_SPLIT = "test"  # the 300 evaluated claims (339 positive qrels) live in the test split

# Dense embedder. The brief (§3) names "sentence-transformers/bge-small-en", but
# no such repo exists — BGE models live under the BAAI org. Pair B MUST embed
# queries with this exact model.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOKEN_LIMIT = 512  # bge-small-en context window; texts above this get truncated


def _token_chunks(tokenizer, title: str, abstract: str, limit: int = TOKEN_LIMIT) -> list[str]:
    """Split ``title + abstract`` into overlapping pieces that each fit ``limit`` tokens.

    Windowing is done on the *abstract*'s token ids (precise — not a word
    heuristic), then the title is prepended to every piece so each chunk keeps
    document context. The per-chunk token budget reserves room for the title
    and the model's [CLS]/[SEP] specials, so no chunk is silently truncated.
    """
    title_ids = tokenizer.encode(title, add_special_tokens=False) if title else []
    abstract_ids = tokenizer.encode(abstract, add_special_tokens=False)

    window = max(limit - len(title_ids) - 8, 64)  # 8 ≈ specials + "\n\n" slack
    overlap = min(50, window // 6)
    stride = max(1, window - overlap)

    chunks: list[str] = []
    for start in range(0, len(abstract_ids), stride):
        piece_ids = abstract_ids[start : start + window]
        if not piece_ids:
            break
        piece = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
        chunks.append(f"{title}\n\n{piece}".strip() if title else piece)
        if start + window >= len(abstract_ids):
            break
    return chunks


def load_scifact() -> "pd.DataFrame":
    """Load SciFact (BEIR) via ir_datasets and return a DataFrame ready for chunking.

    Loads the full ``beir/scifact`` corpus (~5,183 abstracts). The test-split
    claims used for the gold set are loaded separately in
    :func:`build_gold_from_scifact`.

    Chunking strategy (§6.A Q2/Q5): **one chunk per abstract by default** —
    SciFact abstracts are short and a single chunk preserves the full context a
    claim is judged against. The exception: abstracts whose ``title + abstract``
    exceeds the embedder's 512-token window are split into overlapping pieces
    (:func:`_token_chunks`), because otherwise the embedder would silently drop
    the tail — and ~9% of SciFact abstracts overflow. Splitting only the long
    ones keeps the corpus clean while eliminating evidence loss.

    Returns one row per chunk following the §5 schema minus ``embedding``
    (added later by :func:`embed_chunks`): paper_id, chunk_id, text,
    page_start, page_end, title, authors, year, topic, source. Page columns are
    null (SciFact has no page map); authors/year/topic are null (not in BEIR
    SciFact). chunk_id format is ``scifact:<doc_id>:<i>`` (i=0 for short
    abstracts; i=0,1,... for split ones).
    """
    import ir_datasets
    import pandas as pd
    import transformers
    from transformers import AutoTokenizer

    transformers.logging.set_verbosity_error()  # silence >512-token length notices
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

    dataset = ir_datasets.load("beir/scifact")

    rows = []
    for doc in dataset.docs_iter():
        title = (getattr(doc, "title", "") or "").strip()
        abstract = (doc.text or "").strip()
        full_text = f"{title}\n\n{abstract}".strip() if title else abstract

        if len(tokenizer.encode(full_text)) <= TOKEN_LIMIT:
            texts = [full_text]  # common case: one chunk per abstract
        else:
            texts = _token_chunks(tokenizer, title, abstract)

        for i, text in enumerate(texts):
            rows.append(
                {
                    "paper_id": doc.doc_id,
                    "chunk_id": f"scifact:{doc.doc_id}:{i}",
                    "text": text,
                    "page_start": None,  # SciFact abstracts have no page map
                    "page_end": None,
                    "title": title,
                    "authors": None,
                    "year": None,
                    "topic": None,
                    "source": "scifact",
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "paper_id",
            "chunk_id",
            "text",
            "page_start",
            "page_end",
            "title",
            "authors",
            "year",
            "topic",
            "source",
        ],
    )


ARXIV_META_JSON = DATA_RAW / "arxiv_meta.json"


def download_arxiv_demo(n: int = ARXIV_DEMO_COUNT, query: str = "cat:cs.CL") -> list[Path]:
    """Download a small number of arXiv PDFs into DATA_RAW for the pipeline demo.

    These are NOT used in evaluation — they exist only to prove the PDF parser
    runs end-to-end so D2 doesn't start from zero. Picks the ``n`` most recent
    cs.CL submissions (sorted by submitted_date, descending).

    Also writes ``data/raw_pdfs/arxiv_meta.json`` — a sidecar carrying the
    arXiv-supplied title/authors/year, since those are unreliable inside the
    PDF itself. :func:`parse_arxiv_pdfs` reads it back to fill the §5 schema.
    """
    import json

    import arxiv

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=n,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    paths: list[Path] = []
    meta: list[dict] = []
    for result in client.results(search):
        paper_id = result.get_short_id().replace("/", "_")  # e.g. 2405.01234v1
        filename = f"{paper_id}.pdf"
        result.download_pdf(dirpath=str(DATA_RAW), filename=filename)
        paths.append(DATA_RAW / filename)
        meta.append(
            {
                "paper_id": paper_id,
                "title": (result.title or "").strip(),
                "authors": [a.name for a in result.authors],
                "year": result.published.year if result.published else None,
            }
        )

    ARXIV_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"download_arxiv_demo: downloaded {len(paths)} PDFs to {DATA_RAW}")
    return paths


def _chunk_pages(pages: list[str], chunk_tokens: int, overlap: int) -> list[tuple[str, int, int]]:
    """Sliding-window word chunker that tracks which pages each chunk spans.

    Returns ``(text, page_start, page_end)`` tuples. "Tokens" are approximated
    by whitespace-split words — good enough for the demo chunks (not evaluated)
    and avoids dragging in the SBERT tokenizer just to count.
    """
    words: list[tuple[str, int]] = []  # (word, 1-based page number)
    for page_no, page_text in enumerate(pages, start=1):
        for word in page_text.split():
            words.append((word, page_no))

    if not words:
        return []

    stride = max(1, chunk_tokens - overlap)
    chunks: list[tuple[str, int, int]] = []
    for start in range(0, len(words), stride):
        window = words[start : start + chunk_tokens]
        if not window:
            break
        text = " ".join(word for word, _ in window)
        chunks.append((text, window[0][1], window[-1][1]))
        if start + chunk_tokens >= len(words):
            break  # last window already reached the end
    return chunks


def parse_arxiv_pdfs(pdf_paths: list[Path], chunk_tokens: int = 220, overlap: int = 40) -> "pd.DataFrame":
    """Parse and chunk arXiv PDFs with PyMuPDF, preserving page_start/page_end. source='arxiv-demo'.

    PyMuPDF (fitz) is the speed/fidelity winner for cs.CL papers (§6.A Q3).
    Title/authors/year come from the ``arxiv_meta.json`` sidecar written by
    :func:`download_arxiv_demo`; if it is absent those columns are null.

    A conservative 220-word window (not 300) keeps chunks comfortably under
    bge-small-en's 512 subword-token limit: dense academic text (citations,
    equations, hyphenated terms) expands well past one token per word, so a
    300-word window risks silent truncation at embed time. :func:`embed_chunks`
    additionally warns if any chunk still overflows.
    """
    import json

    import fitz
    import pandas as pd

    meta: dict[str, dict] = {}
    if ARXIV_META_JSON.exists():
        for entry in json.loads(ARXIV_META_JSON.read_text(encoding="utf-8")):
            meta[entry["paper_id"]] = entry

    rows = []
    for pdf_path in pdf_paths:
        paper_id = pdf_path.stem
        doc = fitz.open(pdf_path)
        try:
            pages = [page.get_text("text") for page in doc]
        finally:
            doc.close()

        m = meta.get(paper_id, {})
        for i, (text, page_start, page_end) in enumerate(_chunk_pages(pages, chunk_tokens, overlap)):
            rows.append(
                {
                    "paper_id": paper_id,
                    "chunk_id": f"arxiv:{paper_id}:{i}",
                    "text": text,
                    "page_start": page_start,
                    "page_end": page_end,
                    "title": m.get("title"),
                    "authors": m.get("authors"),
                    "year": m.get("year"),
                    "topic": "cs.CL",
                    "source": "arxiv-demo",
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "paper_id",
            "chunk_id",
            "text",
            "page_start",
            "page_end",
            "title",
            "authors",
            "year",
            "topic",
            "source",
        ],
    )


def embed_chunks(df: "pd.DataFrame", model_name: str = EMBED_MODEL) -> "pd.DataFrame":
    """Add 'embedding' column (list[float], 384-dim) using the named SBERT model. Batch=32.

    NOTE (cross-pair, §3/§5): the brief names the embedder
    ``sentence-transformers/bge-small-en``, but no such repo exists — BGE models
    live under the ``BAAI/`` org. We use ``BAAI/bge-small-en-v1.5`` (384-dim,
    same schema). Pair B MUST embed queries with this same model.

    Embeddings are stored **raw** (not L2-normalized) on purpose: ``normalize``
    is a tunable in Pair B's Optuna search space (§6.B), so ingest must not
    pre-decide it — doing so would turn their knob into a no-op.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    # Guardrail (§6.A Q4): warn if any chunk exceeds the model's token limit and
    # would be silently truncated at encode time — most likely for arXiv-demo
    # chunks, since SciFact abstracts are short.
    texts = df["text"].tolist()
    max_len = model.get_max_seq_length()
    n_over = sum(len(model.tokenizer.encode(t, add_special_tokens=True)) > max_len for t in texts)
    if n_over:
        print(f"embed_chunks: WARNING — {n_over}/{len(texts)} chunks exceed "
              f"{max_len} tokens and will be truncated during embedding.")

    embeddings = model.encode(
        texts,
        batch_size=32,  # fits CPU memory
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    out = df.copy()
    out["embedding"] = [emb.astype("float32").tolist() for emb in embeddings]
    return out


def build_gold_from_scifact(split: str = SCIFACT_SPLIT) -> None:
    """Convert SciFact <split>-split claims + qrels into data/gold/qa.jsonl.

    Line format (§5):
        {"qid": claim_id, "question": claim_text, "relevant_chunk_ids": [...], "topic": null}

    The test split carries the 300 manually-judged claims (339 positive qrels).

    Relevance mapping (§6.A Q5): SciFact qrels are document-level. Most
    abstracts are a single chunk, but long ones are split (see
    :func:`load_scifact`), so a relevant doc_id can map to several chunk_ids.
    We mark **all** chunks of a relevant document as relevant — retrieving any
    chunk of the right document counts. The doc→chunk_id mapping is read back
    from ``chunks.parquet`` so qa.jsonl stays consistent with whatever
    splitting load_scifact actually did. Only claims with >=1 positive qrel are
    written (an unjudged claim cannot be scored by Recall@5 / NDCG@5).
    """
    import json
    from collections import defaultdict

    import ir_datasets
    import pandas as pd

    # doc_id -> [chunk_id, ...], taken from the chunks we actually embedded.
    chunks_df = pd.read_parquet(CHUNKS_PARQUET, columns=["paper_id", "chunk_id", "source"])
    scifact_chunks = chunks_df[chunks_df["source"] == "scifact"]
    doc_to_chunks = scifact_chunks.groupby("paper_id")["chunk_id"].apply(list).to_dict()

    dataset = ir_datasets.load(f"beir/scifact/{split}")

    # group positive qrels by claim: claim_id -> [chunk_id, ...]
    relevant: dict[str, list[str]] = defaultdict(list)
    n_missing = 0
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            chunk_ids = doc_to_chunks.get(qrel.doc_id)
            if not chunk_ids:
                n_missing += 1
                continue  # qrel doc not in corpus — should not happen for SciFact
            relevant[qrel.query_id].extend(chunk_ids)

    if n_missing:
        print(f"build_gold_from_scifact: WARNING — {n_missing} qrels had no matching chunk")

    GOLD_JSONL.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with GOLD_JSONL.open("w", encoding="utf-8") as f:
        for query in dataset.queries_iter():
            chunk_ids = relevant.get(query.query_id)
            if not chunk_ids:
                continue  # claim with no positive evidence — not scorable, skip
            line = {
                "qid": query.query_id,
                "question": query.text,
                "relevant_chunk_ids": sorted(set(chunk_ids)),
                "topic": None,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"build_gold_from_scifact: wrote {n_written} claims to {GOLD_JSONL}")


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
