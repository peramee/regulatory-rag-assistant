from pathlib import Path

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from regulatory_rag.ingestion import (
    DocumentIngestionError,
    chunk_pages,
    extract_document,
    ingest_document,
)
from regulatory_rag.models import ChunkingConfig, DocumentPage


def write_pdf(path: Path, texts: list[str], *, encrypted: bool = False) -> None:
    """Create actual PDFs with standard-font text, including optional blank pages."""
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        if not text:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    if encrypted:
        writer.encrypt("password")
    with path.open("wb") as output:
        writer.write(output)


def test_text_ingestion_preserves_unicode_filename_and_bom(tmp_path: Path) -> None:
    path = tmp_path / "régulation.TXT"
    path.write_text("Règle: keep records.\n\nNext rule.", encoding="utf-8-sig")
    chunks = ingest_document(path, ChunkingConfig(chunk_size=100, overlap=0))
    assert len(chunks) == 1
    assert chunks[0].source_document == path.name
    assert chunks[0].page_number is None
    assert chunks[0].text == "Règle: keep records.\n\nNext rule."
    assert set(chunks[0].model_dump()) == {"chunk_id", "source_document", "page_number", "text"}


@pytest.mark.parametrize(
    ("size", "overlap", "expected"),
    [
        (4, 2, ["abcd", "cdef", "efgh", "ghij"]),
        (4, 0, ["abcd", "efgh", "ij"]),
        (10, 3, ["abcdefghij"]),
        (20, 0, ["abcdefghij"]),
        (1, 0, list("abcdefghij")),
    ],
)
def test_chunk_windows(size: int, overlap: int, expected: list[str]) -> None:
    pages = [DocumentPage(source_document="rule.txt", text="abcdefghij")]
    chunks = chunk_pages(pages, ChunkingConfig(chunk_size=size, overlap=overlap))
    assert [chunk.text for chunk in chunks] == expected
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_paragraph_boundary_and_overlap_preserve_all_text() -> None:
    text = "abcdefgh\n\nijklmnopqrstuv"
    chunks = chunk_pages(
        [DocumentPage(source_document="rule.txt", text=text)],
        ChunkingConfig(chunk_size=14, overlap=3),
    )
    assert chunks[0].text == "abcdefgh\n\n"
    assert all(len(chunk.text) <= 14 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text[-3:] == current.text[:3]
    assert chunks[0].text + "".join(chunk.text[3:] for chunk in chunks[1:]) == text


def test_large_overlap_still_makes_progress() -> None:
    text = "abc\n\ndefghijkl"
    chunks = chunk_pages(
        [DocumentPage(source_document="rule.txt", text=text)],
        ChunkingConfig(chunk_size=6, overlap=5),
    )
    assert chunks[0].text + "".join(chunk.text[5:] for chunk in chunks[1:]) == text


def test_real_pdf_preserves_page_numbers_across_blank_pages(tmp_path: Path) -> None:
    path = tmp_path / "rules.PDF"
    write_pdf(path, ["First page rule.", "", "Third page rule."])
    pages = extract_document(path)
    assert [page.page_number for page in pages] == [1, 2, 3]
    chunks = ingest_document(path, ChunkingConfig(chunk_size=10, overlap=2))
    assert {chunk.page_number for chunk in chunks} == {1, 3}
    assert all(chunk.source_document == "rules.PDF" for chunk in chunks)
    assert "First" in chunks[0].text
    assert all("Third" not in chunk.text for chunk in chunks if chunk.page_number == 1)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_ids_are_stable_and_distinguish_documents_and_positions(tmp_path: Path) -> None:
    path = tmp_path / "rules.txt"
    path.write_text("aaaaaaaa", encoding="utf-8")
    config = ChunkingConfig(chunk_size=4, overlap=0)
    first = ingest_document(path, config)
    assert first == ingest_document(path, config)
    assert first[0].chunk_id != first[1].chunk_id
    other = tmp_path / "other.txt"
    other.write_text("aaaaaaaa", encoding="utf-8")
    assert first[0].chunk_id != ingest_document(other, config)[0].chunk_id
    path.write_text("aaaaaaab", encoding="utf-8")
    assert first[0].chunk_id != ingest_document(path, config)[0].chunk_id


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6), (2.5, 0)])
def test_invalid_chunk_settings(size: int, overlap: int) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=size, overlap=overlap)


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_text_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DocumentIngestionError, match="no extractable text"):
        ingest_document(path)


def test_blank_pdf_rejected(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    write_pdf(path, [""])
    with pytest.raises(DocumentIngestionError, match="OCR"):
        ingest_document(path)


def test_encrypted_pdf_rejected(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    write_pdf(path, ["Private text"], encrypted=True)
    with pytest.raises(DocumentIngestionError, match="Encrypted"):
        ingest_document(path)


def test_malformed_pdf_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(DocumentIngestionError, match="Cannot extract"):
        ingest_document(path)


def test_invalid_text_encoding_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(DocumentIngestionError, match="UTF-8"):
        ingest_document(path)


def test_unsupported_extension_rejected(tmp_path: Path) -> None:
    with pytest.raises(DocumentIngestionError, match="Supported document types"):
        ingest_document(tmp_path / "rules.docx")


def test_missing_file_preserves_filesystem_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_document(tmp_path / "missing.txt")


def test_whitespace_pages_do_not_create_chunks() -> None:
    assert chunk_pages([DocumentPage(source_document="empty.txt", text=" \n\t")]) == []
