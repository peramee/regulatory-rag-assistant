# Regulatory RAG Assistant

A portfolio prototype for grounded answers about regulatory documents.
Currently implemented: local PDF/text extraction, overlapping chunking,
OpenAI-compatible embeddings, persistent Chroma semantic search, a standalone
LLM chat adapter, a grounded RAG question-answering service, a FastAPI API,
a minimal Streamlit frontend, and a local evaluation runner.

## Setup

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For a one-command local launch, copy `.env.example` to `.env`, fill in the
provider settings, load the variables into your shell, and run:

```powershell
python scripts/start.py
```

This starts the API on port 8000 and Streamlit on port 8501. API startup
automatically indexes all PDF and text files below `sources/` (excluding the
`originals` archive) into the configured Chroma collection. Set
`AUTO_INDEX_SOURCES=false` to disable this behavior.

Docker runs the API and Streamlit frontend as separate services:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open http://localhost:8501 after the containers start. The frontend connects to
the API over the internal Compose network at `http://api:8000`. Chroma data
persists in the `chroma-data` volume; source files are read from the local
`sources/` folder. The API is also available at http://localhost:8000.

Run the evaluation against that same containerized API with the optional Compose
profile:

```powershell
docker compose --profile evaluation run --rm evaluation
```

The evaluation service waits for the API health check and writes JSON and CSV
reports to `evaluation/results/` on the host. Pass options such as `--top-k`,
`--dataset`, or `--output-dir` after `evaluation` to customize a run.

On Linux/macOS, activate with `source .venv/bin/activate` instead.

## Ingest a document

A local starter corpus of official ESMA/FCA transaction reporting PDFs is stored
in `sources/transaction_reporting/` when downloaded. The entire folder is ignored
by Git and is not included in a clone. Its local `README.md` and `manifest.json`
identify source URLs, versions, jurisdictions, and ingestion-ready files. It is
not automatically indexed; choose documents before uploading through the API or
using the search CLI. Historical guidance and UK reform material are separated.

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

## Index and search

After installing the package and activating the virtual environment, configure
an embedding endpoint in your shell. For example, in PowerShell:

```powershell
$env:EMBEDDING_API_KEY = "your-api-key"
$env:EMBEDDING_BASE_URL = "https://api.openai.com/v1"
$env:EMBEDDING_MODEL = "text-embedding-3-small"

# Add documents and search. Repeat --index to add more files.
python scripts/search.py "What reporting obligations apply?" --index data/regulation.pdf --index data/notes.txt

# Search the saved index without extracting or embedding documents again.
python scripts/search.py "What reporting obligations apply?"
```

Use `export NAME=value` on Linux/macOS. `.env.example` documents the settings;
the demo reads environment variables and does not automatically load `.env`.
`OPENAI_API_KEY` is accepted as a fallback. The URL and model above are defaults;
an OpenAI-compatible local endpoint can use its own model name and omit the key.
The base URL should include the API prefix (usually `/v1`), not `/embeddings`.

The adapter sends document text and questions to the configured endpoint; Chroma
stores the resulting vectors, passage text, filenames, and page metadata locally.
No live API calls are made by tests. Credentials never enter the index metadata.

Options:

- `--index FILE`: ingest and upsert a file before searching; repeat for multiple files.
- `--db PATH`: persistence directory (default `data/chroma`).
- `--collection NAME`: collection name (default `regulatory_documents`).
- `--top-k N`: maximum passages returned (default 5).
- `--chunk-size N`, `--overlap N`: character settings used when indexing.

Output includes passage text, chunk ID, filename, page (`n/a` for text), and cosine
similarity. Scores range from -1 to 1; higher is closer. They are not confidence
probabilities or evidence-sufficiency judgments. An unrelated question can still
return passages. Empty collections return no matches without an embedding call.

Programmatic usage:

```python
import os

from regulatory_rag.ingestion import ingest_document
from regulatory_rag.providers import OpenAICompatibleEmbeddings
from regulatory_rag.retrieval import Retriever
from regulatory_rag.store import ChromaStore

provider = OpenAICompatibleEmbeddings(api_key=os.environ["EMBEDDING_API_KEY"])
store = ChromaStore("data/chroma", embedding_id=provider.embedding_id)
retriever = Retriever(provider, store)
retriever.index(ingest_document("data/regulation.pdf"))
for passage in retriever.search("What reporting obligations apply?", top_k=3):
    print(passage.model_dump())
```

To replace the provider, implement `EmbeddingProvider`: an `embedding_id` property
and `embed(texts)`, returning one vector per input in order. Its identity must
change whenever its embedding model/configuration changes. No base class,
factory, or dependency injection framework is required.

## LLM interface

`LLMProvider.generate(system_prompt, user_prompt) -> str` is a provider-independent
interface. `OpenAICompatibleLLM` implements the non-streaming
[Chat Completions API](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)
using the existing HTTPX dependency. It is independent of retrieval and does not
construct RAG prompts, check evidence, or generate citations.

Set configuration in your shell (PowerShell example):

```powershell
$env:LLM_MODEL = "your-chat-model"
$env:LLM_BASE_URL = "https://api.openai.com/v1"
$env:LLM_API_KEY = "your-api-key"
$env:LLM_TIMEOUT_SECONDS = "60"
$env:LLM_CONNECT_TIMEOUT_SECONDS = "5"
```

`LLM_MODEL` is required: choose a model supported by your endpoint. The base URL
defaults to `https://api.openai.com/v1`. `LLM_API_KEY` takes precedence over the
`OPENAI_API_KEY` fallback; unauthenticated local endpoints may omit both keys.
`.env.example` contains placeholders only. `.env` and `.env.*` files are ignored,
except `.env.example`. Environment values are read when constructing the provider,
not at import time; `.env` files are not automatically loaded.

```python
from regulatory_rag.providers import LLMError, LLMProvider, OpenAICompatibleLLM

llm: LLMProvider = OpenAICompatibleLLM()
try:
    text = llm.generate("Respond concisely.", "Define a reporting deadline.")
    print(text)
except LLMError as error:
    print(error.code, str(error))
```

This example is a plain model call, not a grounded regulatory answer. For explicit
configuration, pass an `LLMConfig` from `regulatory_rag.config` to the constructor.
API keys use Pydantic `SecretStr`, are excluded from configuration serialization
and representation, and are sent only in the authorization header. The adapter
does not log prompts, responses, or credentials.

Defaults are a 5-second connection timeout and 60-second read/write/pool timeouts.
These are HTTP operation/inactivity limits, not an overall wall-clock deadline.
There are no automatic retries or redirects. HTTP, network, malformed-response,
refusal, and truncation failures raise `LLMError` with a stable `code` and optional
`status_code`; messages omit response bodies and underlying transport details.
Codes are `authentication`, `rate_limit`, `unavailable`, `http_error`, `timeout`,
`connection`, `invalid_response`, `refusal`, and `truncated`. Invalid configuration
and blank user prompts raise `ValueError` (including Pydantic validation errors).

The adapter accepts a completed assistant text response with `finish_reason=stop`.
It rejects truncated or tool-call responses rather than returning partial output.
Streaming, tools, conversation history, model-specific generation parameters, and
models requiring a developer message instead of a system message are not supported
in this initial adapter. Provider compatibility has been tested with mocked HTTP,
not real API calls.

## Grounded question answering

`RAGService.answer_question(question)` retrieves passages, selects a bounded
context, asks the LLM for grounded claims, and returns a Pydantic `RAGResponse`.
Use the embedding and LLM environment settings documented above:

```python
import os

from regulatory_rag.config import RAGConfig
from regulatory_rag.providers import OpenAICompatibleEmbeddings, OpenAICompatibleLLM
from regulatory_rag.retrieval import Retriever
from regulatory_rag.service import RAGService
from regulatory_rag.store import ChromaStore

embeddings = OpenAICompatibleEmbeddings(
    model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
    base_url=os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
)
# Use the same path, collection, and embedding configuration used during indexing.
store = ChromaStore("data/chroma", embedding_id=embeddings.embedding_id)
service = RAGService(
    Retriever(embeddings, store),
    OpenAICompatibleLLM(),
    RAGConfig(top_k=5, max_context_chars=12000),
)
result = service.answer_question("What reporting obligations apply?")
print(result.model_dump_json(indent=2))
```

The response includes:

- `status`: `answered` or `insufficient_evidence`.
- `answer`: individual claims with application-added citation markers, such as
  `Operators must submit an annual report. [C1]`.
- `sources`: only cited chunks, deduplicated in order of first citation. Each has
  `citation_id`, `document`, `page`, `chunk_id`, and verbatim supporting `quotes`.
- `retrieval_scores`: `{chunk_id, score}` entries for all retrieved candidates,
  in retrieval order; scores are cosine similarity, not confidence.
- `retrieved_chunks`: complete retrieved passages and metadata for debugging.
- `context_chunk_ids`: ordered IDs actually supplied to the LLM. The first maps
  to `C1`, the next to `C2`, and so on.
- `refusal_reason`: `no_context`, `model_insufficient`, `invalid_model_output`, or
  `null` for an answer. The last refusal code identifies a validation failure,
  rather than claiming that the corpus itself lacks the necessary information.

Context selection preserves retrieval order, removes duplicate chunk IDs, and
includes only whole chunks within the character budget. Oversized chunks are
skipped, not truncated. The budget counts passage text, not tokens, JSON overhead,
the question, or system instructions. `RAGConfig.min_score` optionally excludes
low-scoring candidates; its default is `None` because thresholds require corpus
calibration and cannot establish evidence sufficiency by themselves.

The grounded system prompt prohibits external knowledge, treats document text as
untrusted evidence, and requires refusal when necessary facts are missing. The
LLM returns JSON claims with a source ID and exact supporting quotation for each
claim. The application validates the schema, source IDs, quote membership, and
answer/refusal consistency. It constructs citations from retrieved metadata,
rather than accepting model-authored filenames or page numbers.
Quote matching tolerates differences in whitespace introduced by PDF layout
(such as line breaks and repeated spaces), but requires unchanged words and
punctuation. Returned citations preserve the original source excerpt. Request
logs include `refusal_reason` to distinguish validation failures from missing evidence.
Invalid model output gets one retry with validation feedback and the same evidence;
both attempts use the same validation rules. Model-declared insufficient evidence
is not retried. Token estimates and LLM timing include both attempts when retried.

No usable context skips the LLM call. A model-declared lack of evidence or invalid
grounded output returns:
`Insufficient evidence in the retrieved documents to answer this question.`
Such responses have no answer citations but retain retrieval diagnostics. Provider
and retrieval failures propagate as errors, not evidence refusals. Invalid model
output is never returned as answer text.

**Grounding limitation:** quote membership and citation validation establish a
traceable reference, not proof that a quotation logically supports a claim. The
LLM still assesses semantic support and evidence sufficiency. Prompt instructions
also cannot guarantee resistance to document prompt injection. Live-model quality
evaluation remains necessary; the tests verify orchestration and deterministic
validation with fake or mocked models. No second model verifier or automatic
repair/retry loop is included in this prototype.

## HTTP API

Install the updated dependencies with `python -m pip install -e ".[dev]"`, then
set the embedding and LLM environment variables described above. Start the API:

```powershell
python -m uvicorn regulatory_rag.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open [Swagger UI](http://127.0.0.1:8000/docs) or
[ReDoc](http://127.0.0.1:8000/redoc). The OpenAPI schema is available at
`/openapi.json`, including multipart upload, query, response, and error schemas.
Providers and storage are constructed during application startup; invalid
configuration fails startup before serving requests. There are no live provider
requests during startup. `.env` files are not loaded automatically.

| Endpoint | Request | Successful response |
| --- | --- | --- |
| `GET /health` | None | `200`, `{"status":"ok"}`; liveness only, no provider calls |
| `POST /documents` | Multipart form with one `file` (.txt or .pdf) | `201` if new chunks are indexed, `200` if existing IDs are reindexed |
| `GET /documents` | None | `200`, `{"documents":[{"document":"rules.pdf","chunk_count":4,"pages":[1,2]}]}` |
| `POST /query` | JSON `{"question":"What reporting obligations apply?","top_k":3}` | `200`, grounded `RAGResponse`, including sources and retrieval diagnostics |

`POST /documents` returns `document`, `chunks_indexed`, and `new_chunks` after
synchronous indexing finishes. Original source basenames are preserved; uploaded
paths are never used as filesystem destinations. Temporary upload files are
removed on success or failure. Repeated identical uploads retain the same chunk
IDs and do not duplicate entries. Different contents with the same filename are
retained as separate chunk versions.

Document inventory is derived from persisted Chroma metadata, so it includes CLI
uploads and survives application restarts. It groups all versions by filename;
two separate documents with the same basename share one inventory entry. `pages`
contains the sorted PDF pages with indexed text, not the document's total page
count; text-only entries have an empty page list.

`top_k` is optional, must be an integer from 1 to 100, and overrides retrieval only
for that request. Questions must contain non-whitespace text and have at most
10,000 characters. Evidence refusal is a valid `200` response with
`status="insufficient_evidence"`, not an HTTP failure.

Example calls (use `curl.exe` instead of `curl` in Windows PowerShell):

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@data/regulation.pdf" http://127.0.0.1:8000/documents
curl http://127.0.0.1:8000/documents
curl -H "Content-Type: application/json" -d '{"question":"What reporting obligations apply?","top_k":3}' http://127.0.0.1:8000/query
```

API storage settings are `CHROMA_PATH` (default `data/chroma`) and
`CHROMA_COLLECTION` (default `regulatory_documents`). `MAX_UPLOAD_BYTES` defaults
to 10,485,760 (10 MiB). The service enforces this per-file limit before extraction
and embedding; multipart parsing/spooling happens first, so this is not a total
HTTP request-body limit. The API shares index settings with the search demo.

Errors use `{"code":"...","detail":"..."}`, with field-level `issues` for
invalid requests. Provider credentials and raw response bodies are not returned.

| Status | Meaning |
| --- | --- |
| `409` | Index/embedding configuration conflict; rebuild using compatible settings |
| `413` | File exceeds the configured upload limit |
| `415` | Unsupported document extension |
| `422` | Invalid JSON/fields, missing file, empty/unreadable document, or unsupported PDF extraction |
| `502` | Embedding failure or unusable LLM response, including upstream credential errors |
| `503` | LLM rate limit/quota exhaustion or provider unavailability |
| `504` | LLM request timeout |
| `500` | Unexpected internal/storage failure; internal details are omitted |

Routes only validate HTTP inputs, call `RAGService`, and format HTTP results.
Synchronous routes run through FastAPI's worker thread handling. A service-level
lock serializes indexing, listing, and question answering within one process, so
reads cannot observe an in-progress index write. Run one Uvicorn worker and avoid
concurrent CLI writes to the same collection. The lock is not a multi-process
coordination mechanism. The API is a local prototype with no authentication.

## Streamlit frontend

Install the optional frontend dependency (`.[dev]` also includes it):

```powershell
python -m pip install -e ".[ui]"
```

Start FastAPI as documented above. In a second terminal, activate the same virtual
environment and start the frontend from the repository root:

```powershell
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Enter a question and select **Submit**. The page shows
the API answer and cited filenames, pages, and supporting quotes beneath it.
Insufficient-evidence responses have a distinct warning. The collapsed
**Retrieved chunks and scores** section shows all retrieved passages, similarity
scores, chunk IDs, and whether each passage was included in the model context.

`API_BASE_URL` defaults to `http://127.0.0.1:8000`. Set it in the frontend's shell
when the API uses a different address, for example:

```powershell
$env:API_BASE_URL = "http://127.0.0.1:8000"
$env:API_TIMEOUT_SECONDS = "120"
```

The frontend has a 5-second connection timeout and a configurable read/write/pool
timeout (120 seconds by default). Connection failures, API errors, and malformed
responses are displayed without a traceback. A failed or blank submission clears
the previous answer so it is not mistaken for a new result. Successful results
remain available across page reruns without repeating the API call.

The Streamlit process needs only the API connection settings, not provider keys.
It calls `POST /query` through `api_client.py` and shares only Pydantic response
schemas with the backend. Retrieval, evidence checks, citation construction, and
LLM calls remain in FastAPI's service layer. API-provided text is displayed as
plain text. Styling uses Streamlit defaults with a centered layout.

Index documents through `POST /documents`, Swagger UI, or the existing search CLI
before asking questions. This minimal frontend does not add upload management,
authentication, chat history, or streaming.

## Evaluation

The tracked [evaluation dataset](/H:/Programming/regulatory-rag-assistant/evaluation/dataset.json)
contains 20 manually authored questions: 16 answerable questions tied to a
specific source PDF and page, plus 4 deliberately unanswerable questions. The
questions use the optional local transaction-reporting corpus described above.

Run an isolated evaluation collection after configuring both embedding and LLM
providers:

```powershell
python scripts/evaluate.py --index-corpus --collection transaction_reporting_evaluation
```

`--index-corpus` indexes each prepared PDF in `sources/transaction_reporting/`,
excluding encrypted originals. Omit it to evaluate an existing collection, or
repeat `--index FILE` to select individual documents. Use a separate collection
for EU, historical UK, or UK reform material when those jurisdictions or dates
must not be mixed.

The runner writes `evaluation/results/results.json` and `results.csv`; that
generated directory is ignored by Git. It also prints aggregate results to the
console. JSON preserves every full per-question result, including retrieved
chunks, sources, answers, errors, timings, and failures. CSV provides one flat
analysis row per question. A provider/retrieval failure is recorded as
`outcome="failed"` and does not stop later cases.

For answerable cases, retrieval hit@k requires a top-k chunk matching the
expected filename **and** PDF page. `document_hit_at_k` is included separately
to distinguish finding the right document on the wrong page. Mean reciprocal
rank uses the rank of the first filename/page match. Refusal cases are excluded
from retrieval metrics and contribute to refusal accuracy instead.

Groundedness is deterministic: an answered response must have citations whose
document, page, chunk ID, and supporting quotes agree with the returned retrieved
chunks, and each citation marker must appear in the answer. This validates
traceability, not whether each quote semantically entails each claim. Add a
reviewed judge or human assessment before treating groundedness as factual-quality
assurance. Retrieval latency measures the service's vector search; total latency
measures the entire answer operation, including locking, prompt construction, and
LLM generation. Failed operations have total latency but may lack retrieval latency.

The dataset reference answers are human-written expected-answer guides, retained
in the output for manual analysis. They are not scored with lexical overlap,
because wording-only scoring is misleading for regulatory answers.

## Structured request logging

The FastAPI application emits one JSON log line for every RAG request. Successful
requests use `rag_request_completed`; retrieval or LLM failures use
`rag_request_failed`. Each record includes a request ID, retrieval/LLM/total
latencies in milliseconds, retrieved-chunk count and scores, model name, refusal
state, and stable error type/code where applicable. The API also returns the
request ID in the `X-Request-ID` response header.

Set `LOG_LEVEL` to control the `regulatory_rag` logger, for example:

```powershell
$env:LOG_LEVEL = "INFO"
```

Logs intentionally exclude questions, prompts, answers, document passages,
provider endpoints, API keys, and exception messages. Token counts are local
character-based estimates (roughly four characters per token); the current chat
adapter does not expose provider-reported usage.

## Design

- `models.py` contains Pydantic page/chunk models and validated chunk settings.
- `ingestion.py` exposes `extract_document`, `chunk_pages`, and `ingest_document`.
- `DocumentChunk` is the shared chunk model; `Chunk` remains a compatibility alias.
- `providers.py` isolates the HTTP embedding adapter and validates responses.
  Embeddings are reordered by response index and checked for count, dimension,
  finite values, and nonzero vectors. Requests have a 30-second timeout.
- The same module exposes `LLMProvider`, `OpenAICompatibleLLM`, and `LLMError`.
  `config.py` validates environment-backed LLM settings. These have no dependency
  on the vector store or retrieval orchestration.
- `generation.py` builds grounded prompts, validates structured model claims,
  and renders answers with citations. `service.py` coordinates retrieval and
  generation. `RAGConfig` provides context-selection limits; `RAGResponse` keeps
  the answer and retrieval trace together without depending on an API framework.
- `api.py` defines FastAPI routes, application startup wiring, and HTTP error
  mappings. `RAGService.index_document` handles staged uploads and indexing;
  `RAGService.list_documents` delegates inventory to the Chroma store. API schemas
  live in `models.py` alongside the existing chunk and answer models.
- `streamlit_app.py` presents the question form and API results. `api_client.py`
  handles HTTP transport, response validation, and frontend error messages only.
- `evaluation.py` evaluates the existing `RAGService` without duplicating
  retrieval or generation logic. It writes JSON/CSV results and aggregates
  source/page hit@k, document hit@k, MRR, citation/quote groundedness, refusal
  accuracy, and retrieval/total latency. `scripts/evaluate.py` wires configured
  providers and an optional evaluation corpus to that evaluator.
- `store.py` handles persistent Chroma storage with explicit embeddings and cosine
  distance. It disables Chroma's automatic embedding function. Search reports
  `1 - distance` as similarity, consistent with the configured
  [Chroma cosine index](https://docs.trychroma.com/docs/collections/configure).
- `retrieval.py` embeds documents in batches of 64 and questions individually.
  All document embeddings are validated before storage writes begin. Chroma
  writes use batches of at most 100 records.
- The collection stores its embedding identity, dimension, and schema version.
  Incompatible models/dimensions are rejected. Use a new `--collection` or `--db`
  and reindex when changing embedding models or endpoints.
- Re-indexing identical chunks updates the same IDs without duplicating records,
  but currently still recomputes embeddings. Changed files produce new chunk IDs;
  old versions are retained. Use a fresh collection for a clean corpus rebuild.
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

Tests generate small real PDF fixtures, mock HTTP embedding responses, and use
fixed fake vectors with real temporary Chroma databases. They cover persistence
across processes, metadata, ranking, invalid inputs, the CLI, grounded answers,
insufficient evidence, citation/quote validation, and an end-to-end RAG flow with
mocked HTTP providers. FastAPI integration tests exercise uploads, persistent
inventory, citations, request validation, error mapping, and OpenAPI using a real
temporary Chroma store with fake providers. Streamlit AppTest checks question
submission, citations, insufficient-evidence rendering, and error recovery;
HTTP-client tests mock the API transport. No credentials, network access, or
embedding model downloads are needed.
Evaluation tests cover dataset validation, source/page ranking, groundedness,
refusal accuracy, failure preservation, aggregate metrics, and JSON/CSV reports.

## Limitations

No OCR: image-only PDFs cannot be ingested. Blank/image-only pages are skipped
when other pages contain text, so mixed scanned/text PDFs may be incomplete.
Complex tables, columns, and reading order depend on PDF extraction quality.
Character windows may split sentences or words. Documents are processed in
memory; the API applies a per-file upload limit, while direct ingestion does not.
Original files are not copied into the index.
The index persists extracted chunks, metadata, and vectors only.

The prototype assumes one application process; service operations are serialized.
Chroma batch writes are not a whole-document transaction: a storage failure can leave partial
data; retrying the same chunks is idempotent. There is no automatic retry for
provider failures, token-aware batching, or document-version deletion workflow.
Real embedding quality and provider compatibility require a configured endpoint
and are not established by fake-vector tests. Local storage can be used without
a database server; a hosted embedding endpoint still needs network access.
