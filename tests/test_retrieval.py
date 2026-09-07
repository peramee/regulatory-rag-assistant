import os
import subprocess
import sys
from pathlib import Path

import pytest

from regulatory_rag.ingestion import ingest_document
from regulatory_rag.models import Chunk, DocumentChunk
from regulatory_rag.providers import EmbeddingError
from regulatory_rag.retrieval import Retriever
from regulatory_rag.store import ChromaStore


class FakeEmbeddings:
    """Fixed vectors test ranking mechanics, not real model semantic quality."""

    embedding_id = "test-model-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0] if "report" in text.lower() else [0.0, 1.0] for text in texts]


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(tmp_path / "index", embedding_id="test-model-v1")


@pytest.fixture
def chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="reporting",
            source_document="rules.pdf",
            page_number=3,
            text="Submit annual reports.",
        ),
        DocumentChunk(
            chunk_id="retention",
            source_document="records.txt",
            text="Retain records for five years.",
        ),
    ]


def test_semantic_search_metadata_scores_and_upsert(
    store: ChromaStore, chunks: list[DocumentChunk]
):
    provider = FakeEmbeddings()
    retriever = Retriever(provider, store, batch_size=1)
    assert retriever.index(chunks) == 2
    assert provider.calls == [[chunk.text] for chunk in chunks]
    assert retriever.index(chunks) == 2
    assert store.count() == 2
    results = retriever.search("What reporting obligations apply?", top_k=10)
    assert len(results) == 2
    assert results[0].chunk_id == "reporting"
    assert results[0].text == "Submit annual reports."
    assert results[0].source_document == "rules.pdf"
    assert results[0].page_number == 3
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)
    assert results[1].page_number is None
    assert len(retriever.search("reporting", top_k=1)) == 1


def test_text_ingestion_to_retrieval(tmp_path: Path, store: ChromaStore) -> None:
    path = tmp_path / "rules.txt"
    path.write_text("Reporting is mandatory.", encoding="utf-8")
    chunks = ingest_document(path)
    assert isinstance(chunks[0], DocumentChunk)
    assert Chunk is DocumentChunk
    retriever = Retriever(FakeEmbeddings(), store)
    retriever.index(chunks)
    result = retriever.search("reporting")[0]
    assert result.chunk_id == chunks[0].chunk_id
    assert result.source_document == path.name


def test_persistence_across_processes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    code = (
        "import sys; from regulatory_rag.store import ChromaStore; "
        "from regulatory_rag.models import DocumentChunk; "
        "s=ChromaStore(sys.argv[1], embedding_id='test-model-v1'); "
        "s.upsert_chunks([DocumentChunk(chunk_id='saved', source_document='rules.pdf', "
        "page_number=2, text='Annual reports')], [[1.0,0.0]])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "persisted")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    reopened = ChromaStore(tmp_path / "persisted", embedding_id="test-model-v1")
    assert reopened.count() == 1
    assert reopened.search([1.0, 0.0])[0].page_number == 2


def test_empty_index_does_not_embed(store: ChromaStore) -> None:
    provider = FakeEmbeddings()
    retriever = Retriever(provider, store)
    assert retriever.index([]) == 0
    assert retriever.search("reporting") == []
    assert provider.calls == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(store: ChromaStore, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        Retriever(FakeEmbeddings(), store).search("reporting", top_k=top_k)


def test_blank_question(store: ChromaStore) -> None:
    with pytest.raises(ValueError, match="blank"):
        Retriever(FakeEmbeddings(), store).search(" \n")


def test_model_mismatch_is_rejected(tmp_path: Path, store: ChromaStore) -> None:
    with pytest.raises(ValueError, match="Incompatible"):
        ChromaStore(tmp_path / "index", embedding_id="different-model")
    provider = FakeEmbeddings()
    provider.embedding_id = "different-model"
    with pytest.raises(ValueError, match="does not match"):
        Retriever(provider, store)


def test_dimension_mismatch_is_rejected(store: ChromaStore, chunks: list[DocumentChunk]) -> None:
    store.upsert_chunks(chunks, [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="dimension changed"):
        store.search([1, 0, 0])
    with pytest.raises(ValueError, match="dimension changed"):
        store.upsert_chunks(chunks, [[1, 0, 0], [0, 1, 0]])
    assert store.count() == 2


def test_failed_embedding_batch_does_not_write(store: ChromaStore, chunks: list[DocumentChunk]):
    class FailingEmbeddings(FakeEmbeddings):
        def embed(self, texts: list[str]) -> list[list[float]]:
            if self.calls:
                raise EmbeddingError("Provider unavailable")
            return super().embed(texts)

    with pytest.raises(EmbeddingError):
        Retriever(FailingEmbeddings(), store, batch_size=1).index(chunks)
    assert store.count() == 0


def test_duplicate_ids_rejected_before_embedding(store: ChromaStore, chunks: list[DocumentChunk]):
    provider = FakeEmbeddings()
    with pytest.raises(ValueError, match="unique"):
        Retriever(provider, store).index([chunks[0], chunks[0]])
    assert provider.calls == []


def test_invalid_vectors_do_not_write(store: ChromaStore, chunks: list[DocumentChunk]):
    with pytest.raises(EmbeddingError):
        store.upsert_chunks(chunks, [[1, 0], [0, 0]])
    assert store.count() == 0
