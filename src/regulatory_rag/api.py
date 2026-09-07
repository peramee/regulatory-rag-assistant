"""HTTP schemas, error mapping, and thin routes over the application service."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from regulatory_rag.ingestion import DocumentIngestionError
from regulatory_rag.models import (
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    IndexingResponse,
    QueryRequest,
    RAGResponse,
    ValidationIssue,
)
from regulatory_rag.providers import (
    EmbeddingError,
    LLMError,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleLLM,
)
from regulatory_rag.retrieval import Retriever
from regulatory_rag.service import DocumentUploadError, RAGService
from regulatory_rag.store import ChromaStore, IndexCompatibilityError


async def service_error_handler(request: Request, exc: Exception) -> JSONResponse:
    status, code, detail = 500, "internal_error", "The operation failed due to an internal error"
    if isinstance(exc, DocumentUploadError):
        status = {"too_large": 413, "unsupported_type": 415}.get(exc.code, 422)
        code, detail = exc.code, str(exc)
    elif isinstance(exc, DocumentIngestionError):
        status, code, detail = 422, "invalid_document", str(exc)
    elif isinstance(exc, IndexCompatibilityError):
        status, code, detail = 409, "incompatible_index", str(exc)
    elif isinstance(exc, EmbeddingError):
        status, code = 502, "embedding_error"
        detail = "Embedding service failed; check its configuration and availability"
    elif isinstance(exc, LLMError):
        status = {"timeout": 504, "rate_limit": 503, "unavailable": 503}.get(exc.code, 502)
        code = f"llm_{exc.code}"
        detail = {
            "timeout": "LLM service timed out",
            "rate_limit": "LLM rate limit or quota exceeded; try again later",
            "unavailable": "LLM service is unavailable; try again later",
            "authentication": "LLM credentials or permissions are invalid; check configuration",
        }.get(exc.code, "LLM service failed to return a usable response")
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(code=code, detail=detail).model_dump(exclude_none=True),
    )


def create_app(service: RAGService | None = None) -> FastAPI:
    """Wire real providers at startup, or accept a service for local integration tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        nonlocal service
        if service is None:
            embeddings = OpenAICompatibleEmbeddings(
                model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                base_url=os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            )
            llm = OpenAICompatibleLLM()
            store = ChromaStore(
                os.environ.get("CHROMA_PATH", "data/chroma"),
                embedding_id=embeddings.embedding_id,
                collection_name=os.environ.get("CHROMA_COLLECTION", "regulatory_documents"),
            )
            service = RAGService(
                Retriever(embeddings, store),
                llm,
                max_upload_bytes=int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
            )
        yield

    application = FastAPI(
        title="Regulatory RAG Assistant",
        version="0.1.0",
        description="Index PDF/text documents and ask grounded questions with source citations.",
        lifespan=lifespan,
    )

    def active_service() -> RAGService:
        if service is None:
            raise HTTPException(status_code=503, detail="Application service has not started")
        return service

    for error in (
        DocumentUploadError,
        DocumentIngestionError,
        IndexCompatibilityError,
        EmbeddingError,
        LLMError,
        Exception,
    ):
        application.add_exception_handler(error, service_error_handler)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error = ErrorResponse(
            code="invalid_request",
            detail="Request validation failed",
            issues=[
                ValidationIssue(location=list(item["loc"]), message=item["msg"])
                for item in exc.errors()
            ],
        )
        return JSONResponse(status_code=422, content=error.model_dump())

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Check application liveness",
    )
    def health() -> HealthResponse:
        """Simple liveness check; does not call external providers."""
        return HealthResponse()

    @application.get(
        "/documents",
        response_model=DocumentListResponse,
        tags=["Documents"],
        responses={500: {"model": ErrorResponse}},
    )
    def list_documents() -> DocumentListResponse:
        """List indexed source filenames, chunk counts, and indexed PDF pages."""
        return active_service().list_documents()

    @application.post(
        "/documents",
        response_model=IndexingResponse,
        status_code=201,
        tags=["Documents"],
        responses={
            200: {"model": IndexingResponse, "description": "Existing chunks reindexed"},
            409: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def index_document(
        file: Annotated[UploadFile, File(description="A UTF-8 .txt or text-based PDF document")],
        response: Response,
    ) -> IndexingResponse:
        """Synchronously ingest and index one file; repeated identical uploads are idempotent."""
        result = active_service().index_document(file.filename or "", file.file)
        response.status_code = 201 if result.new_chunks else 200
        return result

    @application.post(
        "/query",
        response_model=RAGResponse,
        tags=["Questions"],
        responses={
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def query(body: QueryRequest) -> RAGResponse:
        """Return a grounded answer/citations or an insufficient-evidence result (both HTTP 200)."""
        return active_service().answer_question(body.question, top_k=body.top_k)

    return application


app = create_app()
