import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from regulatory_rag.api import create_app
from regulatory_rag.providers import EmbeddingError, LLMError
from regulatory_rag.retrieval import Retriever
from regulatory_rag.service import RAGService
from regulatory_rag.store import ChromaStore, IndexCompatibilityError


class FakeEmbeddings:
    embedding_id = "api-test-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "report" in text.lower() else [0.0, 1.0] for text in texts]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(user_prompt)
        evidence = json.loads(user_prompt)["evidence"][0]
        return json.dumps(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": evidence["text"],
                        "evidence": [
                            {"source_id": evidence["source_id"], "quote": evidence["text"]}
                        ],
                    }
                ],
            }
        )


@pytest.fixture
def service(tmp_path: Path):
    store = ChromaStore(tmp_path / "chroma", embedding_id="api-test-v1")
    return RAGService(Retriever(FakeEmbeddings(), store), FakeLLM())


@pytest.fixture
def client(service):
    with TestClient(create_app(service), raise_server_exceptions=False) as test_client:
        yield test_client


def upload(client, filename="rules.txt", text="Operators must submit annual reports."):
    return client.post("/documents", files={"file": (filename, text.encode(), "text/plain")})


def pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    page = writer.add_blank_page(width=612, height=792)
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
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (Annual reports are mandatory.) Tj ET")
    page[NameObject("/Contents")] = stream
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_health_and_empty_inventory(client, service):
    assert client.get("/health").json() == {"status": "ok"}
    result = client.get("/documents")
    assert result.status_code == 200
    assert result.json() == {"documents": []}
    assert service.llm.calls == []


def test_upload_list_and_query_end_to_end(client):
    result = upload(client)
    assert result.status_code == 201
    assert result.json() == {"document": "rules.txt", "chunks_indexed": 1, "new_chunks": 1}
    assert client.get("/documents").json() == {
        "documents": [{"document": "rules.txt", "chunk_count": 1, "pages": []}]
    }
    query = client.post("/query", json={"question": "What reporting obligations apply?"})
    assert query.status_code == 200
    body = query.json()
    assert body["status"] == "answered"
    assert body["answer"] == "Operators must submit annual reports. [C1]"
    assert body["sources"][0]["document"] == "rules.txt"
    assert body["sources"][0]["page"] is None
    assert body["sources"][0]["chunk_id"] == body["retrieved_chunks"][0]["chunk_id"]
    assert body["retrieval_scores"][0]["score"] == pytest.approx(1.0)


def test_pdf_upload_and_page_citations(client):
    result = client.post(
        "/documents", files={"file": ("rules.pdf", pdf_bytes(), "application/pdf")}
    )
    assert result.status_code == 201
    assert client.get("/documents").json()["documents"] == [
        {"document": "rules.pdf", "chunk_count": 1, "pages": [2]}
    ]
    query = client.post("/query", json={"question": "Reporting?"})
    assert query.status_code == 200
    assert query.json()["sources"][0]["page"] == 2


def test_duplicate_upload_is_idempotent(client):
    assert upload(client).status_code == 201
    second = upload(client)
    assert second.status_code == 200
    assert second.json()["new_chunks"] == 0
    assert client.get("/documents").json()["documents"][0]["chunk_count"] == 1


def test_changed_versions_are_grouped_by_filename(client):
    upload(client)
    assert upload(client, text="Reports are due every month.").status_code == 201
    assert client.get("/documents").json()["documents"] == [
        {"document": "rules.txt", "chunk_count": 2, "pages": []}
    ]


def test_top_k_override_does_not_modify_service_defaults(client, service):
    upload(client)
    upload(client, filename="retention.txt", text="Keep records for five years.")
    first = client.post("/query", json={"question": "Reporting?", "top_k": 1})
    assert first.status_code == 200
    assert len(first.json()["retrieved_chunks"]) == 1
    second = client.post("/query", json={"question": "Reporting?"})
    assert len(second.json()["retrieved_chunks"]) == 2
    assert service.config.top_k == 5


def test_insufficient_evidence_is_successful_query_response(client, service):
    result = client.post("/query", json={"question": "Reporting?"})
    assert result.status_code == 200
    assert result.json()["status"] == "insufficient_evidence"
    assert result.json()["sources"] == []
    assert service.llm.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": " \n"},
        {"question": 123},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 101},
        {"question": "q", "top_k": True},
        {"question": "q", "top_k": "2"},
        {"question": "q", "unexpected": "value"},
        {"question": "x" * 10001},
    ],
)
def test_invalid_query_body_returns_useful_422(client, body):
    result = client.post("/query", json=body)
    assert result.status_code == 422
    assert result.json()["code"] == "invalid_request"
    assert result.json()["issues"][0]["location"][0] == "body"
    assert "input" not in result.json()["issues"][0]


def test_malformed_json_and_missing_file(client):
    result = client.post("/query", content="{", headers={"content-type": "application/json"})
    assert result.status_code == 422
    assert result.json()["code"] == "invalid_request"
    result = client.post("/documents")
    assert result.status_code == 422
    assert result.json()["issues"][0]["location"] == ["body", "file"]


@pytest.mark.parametrize(
    "filename,content,status,code",
    [
        ("rules.docx", b"text", 415, "unsupported_type"),
        ("rules.txt", b"", 422, "invalid_document"),
        ("rules.txt", b"\xff", 422, "invalid_document"),
        ("rules.pdf", b"not a PDF", 422, "invalid_document"),
    ],
)
def test_invalid_uploads_are_not_indexed(client, filename, content, status, code):
    result = client.post("/documents", files={"file": (filename, content)})
    assert result.status_code == status
    assert result.json()["code"] == code
    assert client.get("/documents").json() == {"documents": []}


def test_upload_size_limit(client, service):
    service.max_upload_bytes = 10
    result = upload(client, text="x" * 11)
    assert result.status_code == 413
    assert result.json()["code"] == "too_large"
    assert "10-byte" in result.json()["detail"]
    assert client.get("/documents").json() == {"documents": []}
    assert upload(client, text="x" * 10).status_code == 201


@pytest.mark.parametrize(
    "filename", ["../../rules.txt", "folder/rules.txt", r"C:\fakepath\rules.txt"]
)
def test_upload_paths_are_reduced_to_source_basename(client, filename):
    result = upload(client, filename=filename)
    assert result.status_code == 201
    assert result.json()["document"] == "rules.txt"
    assert client.get("/documents").json()["documents"][0]["document"] == "rules.txt"


@pytest.mark.parametrize(
    "code,status",
    [
        ("timeout", 504),
        ("rate_limit", 503),
        ("unavailable", 503),
        ("authentication", 502),
        ("invalid_response", 502),
    ],
)
def test_llm_errors_map_to_upstream_http_statuses(client, service, monkeypatch, code, status):
    upload(client)

    def fail(*args):
        raise LLMError("sensitive provider details", code=code)

    monkeypatch.setattr(service.llm, "generate", fail)
    result = client.post("/query", json={"question": "Reporting?"})
    assert result.status_code == status
    assert result.json()["code"] == f"llm_{code}"
    assert "sensitive" not in result.text


def test_embedding_error_leaves_index_empty(client, service, monkeypatch):
    def fail(texts):
        raise EmbeddingError("sensitive details")

    monkeypatch.setattr(service.retriever.provider, "embed", fail)
    result = upload(client)
    assert result.status_code == 502
    assert result.json()["code"] == "embedding_error"
    assert "sensitive" not in result.text
    assert client.get("/documents").json() == {"documents": []}


def test_index_conflict_and_unexpected_error_responses(client, service, monkeypatch):
    def conflict(question, *, top_k):
        raise IndexCompatibilityError("Embedding dimension changed; reindex documents")

    monkeypatch.setattr(service, "answer_question", conflict)
    result = client.post("/query", json={"question": "Reporting?"})
    assert result.status_code == 409
    assert result.json()["code"] == "incompatible_index"

    def fail():
        raise RuntimeError("private internal file path")

    monkeypatch.setattr(service, "list_documents", fail)
    result = client.get("/documents")
    assert result.status_code == 500
    assert result.json()["code"] == "internal_error"
    assert "private" not in result.text


def test_openapi_and_interactive_docs(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/documents", "/query", "/health"}
    upload_schema = schema["paths"]["/documents"]["post"]
    assert "multipart/form-data" in upload_schema["requestBody"]["content"]
    assert set(upload_schema["responses"]) >= {"200", "201", "413", "415", "422"}
    assert schema["components"]["schemas"]["QueryRequest"]["properties"]["top_k"]
    assert "sources" in schema["components"]["schemas"]["RAGResponse"]["properties"]


def test_listing_survives_new_service_instance(client, tmp_path):
    upload(client)
    store = ChromaStore(tmp_path / "chroma", embedding_id="api-test-v1")
    reopened = RAGService(Retriever(FakeEmbeddings(), store), FakeLLM())
    with TestClient(create_app(reopened)) as second:
        assert second.get("/documents").json()["documents"][0]["document"] == "rules.txt"


def test_default_startup_uses_environment_without_live_provider_calls(tmp_path, monkeypatch):
    for name in ("LLM_API_KEY", "EMBEDDING_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_MODEL", "local-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "startup"))
    monkeypatch.setenv("CHROMA_COLLECTION", "startup_test")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/documents").json() == {"documents": []}
        assert (
            client.post("/query", json={"question": "Reporting?"}).json()["status"]
            == "insufficient_evidence"
        )
