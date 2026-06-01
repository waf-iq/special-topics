"""D2-A1 runner — kick off an arxiv batch ingest into chunks.parquet.

Usage:
    python scripts/ingest_arxiv_batch.py --max 150 --query "cat:cs.CL"

Resumable: re-running picks up paper_ids not yet in the parquet. Errors are
captured in data/raw_pdfs/arxiv_batch_errors.json (non-fatal).

For an unattended ~3-4h run, redirect stdout/stderr to a file you can tail:

    python scripts/ingest_arxiv_batch.py --max 150 \
        > data/raw_pdfs/arxiv_batch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys

from csai415.ingest import ingest_arxiv_batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max", type=int, default=150,
                        help="target number of NEW arxiv papers to add (default 150)")
    parser.add_argument("--query", type=str, default="cat:cs.CL",
                        help='arxiv search query (default "cat:cs.CL")')
    parser.add_argument("--sleep", type=float, default=5.0,
                        help="seconds to sleep between downloads (default 5)")
    parser.add_argument("--chunk-tokens", type=int, default=220,
                        help="words per chunk window (default 220)")
    parser.add_argument("--overlap", type=int, default=40,
                        help="word overlap between chunks (default 40)")
    args = parser.parse_args()

    print(f"D2-A1 ingest_arxiv_batch start: max={args.max} query={args.query!r} sleep={args.sleep}s", flush=True)
    stats = ingest_arxiv_batch(
        max_results=args.max,
        query=args.query,
        sleep_seconds=args.sleep,
        chunk_tokens=args.chunk_tokens,
        overlap=args.overlap,
    )
    print(f"D2-A1 ingest_arxiv_batch done: {json.dumps(stats, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
