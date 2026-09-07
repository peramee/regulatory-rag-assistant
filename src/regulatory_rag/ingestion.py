"""Local document extraction and overlapping, page-preserving chunking."""

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from regulatory_rag.models import Chunk, ChunkingConfig, DocumentPage


class DocumentIngestionError(ValueError):
    """A document is unsupported, unreadable, or has no extractable text."""


def extract_document(path: str | Path) -> list[DocumentPage]:
    """Read UTF-8 text or PDF text, preserving basename and 1-based PDF pages.

    Blank PDF pages remain in the extracted result. Filesystem errors propagate
    so callers can distinguish missing files and access failures from bad input.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        raise DocumentIngestionError("Supported document types are .txt and .pdf")

    if suffix == ".txt":
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentIngestionError("Text documents must use UTF-8 encoding") from exc
        pages = [DocumentPage(source_document=path.name, text=text)]
    else:
        try:
            with path.open("rb") as stream:
                reader = PdfReader(stream)
                if reader.is_encrypted:
                    raise DocumentIngestionError("Encrypted PDFs are not supported")
                pages = [
                    DocumentPage(
                        source_document=path.name,
                        page_number=number,
                        text=page.extract_text() or "",
                    )
                    for number, page in enumerate(reader.pages, start=1)
                ]
        except PyPdfError as exc:
            raise DocumentIngestionError(f"Cannot extract text from PDF: {path.name}") from exc

    if not any(page.text.strip() for page in pages):
        raise DocumentIngestionError(
            "Document contains no extractable text; scanned PDFs require OCR"
        )
    return pages


def chunk_pages(pages: list[DocumentPage], config: ChunkingConfig | None = None) -> list[Chunk]:
    """Split each page independently, retaining exact extracted substrings.

    Prefer paragraph breaks in the latter half of a window when that permits
    forward progress. Otherwise use the character limit. Adjacent windows
    overlap by exactly the configured number of characters. Whitespace-only
    windows are omitted, and no redundant trailing overlap chunk is emitted.
    """
    config = config or ChunkingConfig()
    # Include the whole extracted document so edits invalidate its chunk IDs.
    identity = json.dumps([page.model_dump() for page in pages], ensure_ascii=False, sort_keys=True)
    document_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    chunks: list[Chunk] = []
    for page_index, page in enumerate(pages):
        start = 0
        while start < len(page.text):
            end = min(start + config.chunk_size, len(page.text))
            if end < len(page.text):
                earliest = start + max(config.overlap + 1, config.chunk_size // 2)
                boundary = page.text.rfind("\n\n", earliest, end)
                if boundary != -1:
                    end = boundary + 2
            text = page.text[start:end]
            if text.strip():
                identity = f"{document_hash}:{page_index}:{start}:{end}"
                chunks.append(
                    Chunk(
                        chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                        source_document=page.source_document,
                        page_number=page.page_number,
                        text=text,
                    )
                )
            if end == len(page.text):
                break
            start = end - config.overlap
    return chunks


def ingest_document(
    path: str | Path, config: ChunkingConfig | None = None, *, source_document: str | None = None
) -> list[Chunk]:
    """Extract and chunk one local document without persistence or network calls."""
    pages = extract_document(path)
    if source_document is not None:
        pages = [
            DocumentPage(
                source_document=source_document, page_number=page.page_number, text=page.text
            )
            for page in pages
        ]
    return chunk_pages(pages, config)
