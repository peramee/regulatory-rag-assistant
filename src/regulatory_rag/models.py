"""Validated ingestion inputs and outputs."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    """Character limits; overlap must be smaller than chunk size."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    chunk_size: int = Field(default=1000, gt=0)
    overlap: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class DocumentPage(BaseModel):
    """Extracted text from a PDF page or an entire text file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_document: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    text: str


class DocumentChunk(DocumentPage):
    """A source-located passage with a deterministic identifier."""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1, pattern=r"\S")


# Preserve the name used by the existing ingestion API.
Chunk = DocumentChunk


class SearchResult(DocumentChunk):
    """Retrieved passage; cosine similarity is higher for closer matches."""

    score: float = Field(ge=-1, le=1, allow_inf_nan=False)


class SourceCitation(BaseModel):
    """Trusted source metadata and verbatim support for generated claims."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_id: str
    document: str
    page: int | None = Field(default=None, ge=1)
    chunk_id: str
    quotes: list[str]


class RetrievalScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    score: float = Field(ge=-1, le=1, allow_inf_nan=False)


class RAGResponse(BaseModel):
    """Answer plus citations and the evidence-selection trace, including on refusal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["answered", "insufficient_evidence"]
    answer: str
    sources: list[SourceCitation]
    retrieval_scores: list[RetrievalScore]
    retrieved_chunks: list[SearchResult]
    context_chunk_ids: list[str]
    refusal_reason: Literal["no_context", "model_insufficient", "invalid_model_output"] | None = (
        None
    )
