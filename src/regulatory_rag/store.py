"""Explicit-vector Chroma persistence; no implicit model downloads or embedding."""

from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.types import Metadata
from chromadb.config import Settings

from regulatory_rag.models import DocumentChunk, IndexedDocument, SearchResult
from regulatory_rag.providers import validate_embeddings


class IndexCompatibilityError(ValueError):
    """The selected index and embedding configuration are incompatible."""


class ChromaStore:
    def __init__(
        self,
        path: str | Path,
        *,
        embedding_id: str,
        collection_name: str = "regulatory_documents",
    ) -> None:
        if not embedding_id.strip():
            raise ValueError("embedding_id must not be blank")
        self.embedding_id = embedding_id
        self._client = chromadb.PersistentClient(
            path=str(Path(path).resolve()), settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_id": embedding_id, "schema_version": 1},
        )
        metadata = self._collection.metadata or {}
        if metadata.get("embedding_id") != embedding_id or metadata.get("schema_version") != 1:
            raise IndexCompatibilityError(
                "Incompatible index; use a new collection and reindex documents"
            )
        hnsw = self._collection.configuration.get("hnsw")
        if not hnsw or hnsw.get("space") != "cosine":
            raise IndexCompatibilityError("The index must use cosine distance")

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[IndexedDocument]:
        """Read metadata only, grouping all indexed versions by filename."""
        counts: dict[str, int] = {}
        pages: dict[str, set[int]] = {}
        for offset in range(0, self.count(), 1000):
            records = self._collection.get(limit=1000, offset=offset, include=["metadatas"])
            for metadata in records["metadatas"] or []:
                document = metadata["source_document"]
                if not isinstance(document, str):
                    raise ValueError("Index contains invalid document metadata")
                counts[document] = counts.get(document, 0) + 1
                document_pages = pages.setdefault(document, set())
                page = metadata.get("page_number")
                if isinstance(page, int):
                    document_pages.add(page)
        return [
            IndexedDocument(document=name, chunk_count=counts[name], pages=sorted(pages[name]))
            for name in sorted(counts)
        ]

    def _validate_dimension(self, dimension: int) -> None:
        stored = (self._collection.metadata or {}).get("dimension")
        if stored is not None and stored != dimension:
            raise IndexCompatibilityError(
                "Embedding dimension changed; use a new collection and reindex"
            )

    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
        if not chunks and not vectors:
            return 0
        dimension = validate_embeddings(vectors, len(chunks))
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique within an indexing request")
        self._validate_dimension(dimension)
        metadata = dict(self._collection.metadata or {})
        if "dimension" not in metadata:
            metadata["dimension"] = dimension
            self._collection.modify(metadata=metadata)
        batch_size = min(100, self._client.get_max_batch_size())
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            metadatas: list[Metadata] = []
            for chunk in batch:
                item: dict[str, str | int] = {"source_document": chunk.source_document}
                # Chroma metadata cannot represent None; omission means a text file.
                if chunk.page_number is not None:
                    item["page_number"] = chunk.page_number
                metadatas.append(item)
            batch_vectors: list[Sequence[float] | Sequence[int]] = [
                vector for vector in vectors[start : start + batch_size]
            ]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=metadatas,
                embeddings=batch_vectors,
            )
        return len(chunks)

    def search(self, vector: list[float], *, top_k: int = 5) -> list[SearchResult]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        dimension = validate_embeddings([vector], 1)
        self._validate_dimension(dimension)
        count = self.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=vector,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents, metadatas, distances = (
            result["documents"],
            result["metadatas"],
            result["distances"],
        )
        if documents is None or metadatas is None or distances is None:
            raise ValueError("Index is missing passage data; reindex documents")
        return [
            SearchResult.model_validate(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source_document": metadata["source_document"],
                    "page_number": metadata.get("page_number"),
                    # Chroma returns cosine distance. Clamp floating-point roundoff.
                    "score": max(-1.0, min(1.0, 1.0 - distance)),
                }
            )
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0], documents[0], metadatas[0], distances[0], strict=True
            )
        ]
