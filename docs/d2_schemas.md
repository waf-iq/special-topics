# D2 Storage Schemas

## MongoDB (`csai415` database)

### `papers` collection

One document per unique paper. `_id` is the `paper_id` from `chunks.parquet`.

| Field | Type | Description |
|---|---|---|
| `_id` | string | Paper ID (e.g. `"4983"` for SciFact, `"2405.01234v1"` for arXiv) |
| `title` | string | Paper title |
| `authors` | array[string] | Author names (empty `[]` for SciFact) |
| `year` | int \| null | Publication year (null for SciFact) |
| `venue` | string \| null | Venue name (null for all current sources) |
| `topics` | array[string] | Full arXiv categories (e.g. `["cs.CL", "cs.AI"]`); empty `[]` for SciFact |
| `source` | string | `"scifact"`, `"arxiv"`, or `"arxiv-demo"` |
| `pdf_path` | string \| null | Local path to PDF (null for SciFact) |
| `ingested_at` | datetime | UTC timestamp of seed time |

**Indexes:** `authors`, `year`

Example document:
```json
{
  "_id": "2405.01234v1",
  "title": "Attention Is All You Need (Revisited)",
  "authors": ["Alice Smith", "Bob Jones"],
  "year": 2024,
  "venue": null,
  "topics": ["cs.CL", "cs.AI"],
  "source": "arxiv",
  "pdf_path": "data/raw_pdfs/2405.01234v1.pdf",
  "ingested_at": "2026-06-01T12:00:00Z"
}
```

### `chunks` collection

One document per text chunk. `_id` is the `chunk_id` from `chunks.parquet`.

| Field | Type | Description |
|---|---|---|
| `_id` | string | Chunk ID (e.g. `"scifact:4983:0"`, `"arxiv:2405.01234v1:3"`) |
| `paper_id` | string | FK to `papers._id` |
| `text` | string | Chunk text content |
| `position` | int | Positional index within the seed batch |
| `page_start` | int \| null | First page of chunk (null for SciFact) |
| `page_end` | int \| null | Last page of chunk (null for SciFact) |
| `source` | string | `"scifact"`, `"arxiv"`, or `"arxiv-demo"` |

**Indexes:** `paper_id`

Example document:
```json
{
  "_id": "scifact:4983:0",
  "paper_id": "4983",
  "text": "VEGF signaling promotes angiogenesis in tumor microenvironments...",
  "position": 0,
  "page_start": null,
  "page_end": null,
  "source": "scifact"
}
```

---

## Qdrant (`chunks_bge384` collection)

| Setting | Value |
|---|---|
| Collection name | `chunks_bge384` |
| Vector size | 384 |
| Distance | Cosine |

### Point structure

| Field | Location | Type | Description |
|---|---|---|---|
| `id` | point ID | int | Parquet row index (0..n-1), matches `corpus_idx` in `HybridRetriever` |
| vector | vector | float32[384] | BGE-small-en-v1.5 embedding |
| `chunk_id` | payload | string | Chunk ID (e.g. `"scifact:4983:0"`) |
| `paper_id` | payload | string | Paper ID |
| `source` | payload | string | `"scifact"`, `"arxiv"`, or `"arxiv-demo"` |
| `page_start` | payload | int \| null | First page of chunk |
| `page_end` | payload | int \| null | Last page of chunk |
| `title` | payload | string | Paper title |
| `text` | payload | string | Full chunk text (enables single-store `/search` responses) |

**Eval-time filter:** `source == "scifact"` ensures SciFact qrels remain valid when querying a mixed collection.

Example point (JSON-like):
```json
{
  "id": 42,
  "vector": [0.0123, -0.0456, ...],
  "payload": {
    "chunk_id": "scifact:4983:0",
    "paper_id": "4983",
    "source": "scifact",
    "page_start": null,
    "page_end": null,
    "title": "VEGF signaling in tumor angiogenesis",
    "text": "VEGF signaling promotes angiogenesis in tumor microenvironments..."
  }
}
```
