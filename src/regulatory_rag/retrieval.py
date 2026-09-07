"""Small orchestration layer joining replaceable embeddings and local search."""

from regulatory_rag.models import DocumentChunk, SearchResult
from regulatory_rag.providers import EmbeddingProvider, validate_embeddings
from regulatory_rag.store import ChromaStore


class Retriever:
    def __init__(
        self, provider: EmbeddingProvider, store: ChromaStore, *, batch_size: int = 64
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if provider.embedding_id != store.embedding_id:
            raise ValueError("Provider does not match the index embedding model")
        self.provider = provider
        self.store = store
        self.batch_size = batch_size

    def index(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique within an indexing request")
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            embedded = self.provider.embed([chunk.text for chunk in batch])
            validate_embeddings(embedded, len(batch))
            vectors.extend(embedded)
        # Validate all batches before writing so provider failures leave no partial index.
        return self.store.upsert_chunks(chunks, vectors)

    def search(self, question: str, *, top_k: int = 5) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("Question must not be blank")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.store.count() == 0:
            return []
        vectors = self.provider.embed([question])
        validate_embeddings(vectors, 1)
        return self.store.search(vectors[0], top_k=top_k)
