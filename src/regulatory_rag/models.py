"""Validated ingestion inputs and outputs."""

from typing import Self

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
