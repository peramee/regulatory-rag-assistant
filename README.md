# Regulatory RAG Assistant

A portfolio prototype for grounded answers about regulatory documents.
Currently implemented: local PDF/text extraction, overlapping chunking,
OpenAI-compatible embeddings, persistent Chroma semantic search, and a standalone
LLM chat adapter. Grounded RAG answer generation, FastAPI, evaluation, and Docker
are future work.

## Setup

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On Linux/macOS, activate with `source .venv/bin/activate` instead.

## Ingest a document

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
across processes, metadata, ranking, invalid inputs, and the CLI. No credentials,
network access, or embedding model downloads are needed.

## Limitations

No OCR: image-only PDFs cannot be ingested. Blank/image-only pages are skipped
when other pages contain text, so mixed scanned/text PDFs may be incomplete.
Complex tables, columns, and reading order depend on PDF extraction quality.
Character windows may split sentences or words. Documents are processed in
memory; there are no upload limits. Original files are not copied into the index.
The index persists extracted chunks, metadata, and vectors only.

The prototype assumes one writer and no concurrent ingestion/search. Chroma batch
writes are not a whole-document transaction: a storage failure can leave partial
data; retrying the same chunks is idempotent. There is no automatic retry for
provider failures, token-aware batching, or document-version deletion workflow.
Real embedding quality and provider compatibility require a configured endpoint
and are not established by fake-vector tests. Local storage can be used without
a database server; a hosted embedding endpoint still needs network access.
