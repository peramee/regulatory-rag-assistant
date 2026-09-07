# Regulatory RAG Assistant

A portfolio prototype for grounded answers about regulatory documents.
Currently implemented: local PDF/text extraction and overlapping chunking.
Embeddings, retrieval, generation, FastAPI, evaluation, and Docker are future work.

## Setup

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On Linux/macOS, activate with `source .venv/bin/activate` instead.

## Ingest a document

```python
from regulatory_rag.ingestion import ingest_document
from regulatory_rag.models import ChunkingConfig

chunks = ingest_document(
    "data/regulation.pdf",
    ChunkingConfig(chunk_size=1000, overlap=200),
)
for chunk in chunks:
    print(chunk.model_dump())
```

Each chunk contains `chunk_id`, `source_document` (original basename),
`page_number` (1-based PDF page, or `None` for text), and `text`.

## Design

- `models.py` contains Pydantic page/chunk models and validated chunk settings.
- `ingestion.py` exposes `extract_document`, `chunk_pages`, and `ingest_document`.
- Text files use UTF-8, with optional BOM. PDFs use pypdf text extraction.
- Chunk size and overlap count characters, not tokens. Size is a maximum;
  paragraph breaks in the latter half of a window are preferred when safe.
- PDF chunks never cross pages. Overlap applies within a page, not between pages.
  Extracted whitespace is retained; whitespace-only windows are skipped.
- IDs hash the extracted document metadata/text and window location. Repeated
  ingestion is deterministic; document edits change IDs. Identical content with
  the same basename is treated as the same document regardless of directory.
- Unsupported formats, non-UTF-8 text, encrypted PDFs, malformed PDFs reported by
  pypdf, and documents without text raise `DocumentIngestionError`. Filesystem
  errors retain their normal Python exception types.

## Checks

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Tests generate small real PDF fixtures locally; no credentials or network needed.

## Limitations

No OCR: image-only PDFs cannot be ingested. Blank/image-only pages are skipped
when other pages contain text, so mixed scanned/text PDFs may be incomplete.
Complex tables, columns, and reading order depend on PDF extraction quality.
Character windows may split sentences or words. Documents are processed in
memory; this module does not enforce upload limits or persist documents/chunks.
