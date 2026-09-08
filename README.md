# Regulatory RAG Assistant

A document question answering RAG prototype for regulatory material, with OpenAI API integration for embeddings and grounded chat generation. It extracts PDF/text documents, indexes their chunks in Chroma, retrieves relevant passages, and generates answers with validated source citations.

This project demonstrates an end-to-end, production-minded RAG workflow: document ingestion, indexing, semantic retrieval, grounded generation, API design, OpenAI API integration, a small user interface, observability, automated evaluation, and reproducible Docker deployment. Every answer exposes its supporting passages, and unsupported questions are refused.

## What this demonstrates

- Designing a modular RAG pipeline with clear boundaries between ingestion, retrieval, generation, and presentation.
- Integrating replaceable OpenAI-compatible embedding and chat providers behind typed interfaces.
- Building a persistent vector index with stable document and chunk metadata.
- Validating model output against a strict JSON schema and source quotations before displaying it.
- Exposing the system through a documented FastAPI service and a separate Streamlit client.
- Evaluating retrieval quality, citation traceability, refusal behavior, failures, and latency through the same API used by the UI.

## Architecture

```text
Browser -> Streamlit (frontend:8501) -> FastAPI (api:8000)
                                      |
                                      +-> Chroma persistent index
                                      +-> embedding via OpenAI API
                                      +-> chat completion via OpenAI API

Evaluation (one-shot container) ------> FastAPI
```

The API owns ingestion, retrieval, generation, and the Chroma volume. Streamlit is an HTTP client and contains no RAG logic. The evaluation service calls the same API used by the frontend. The API and frontend run by default; evaluation is enabled only with its Compose profile.

## Technologies

| Technology | Role in the project |
| --- | --- |
| Python | Application language, chosen for its typing support and mature data/AI ecosystem. |
| FastAPI | HTTP API, request validation, OpenAPI documentation, and consistent error responses. |
| Pydantic | Strict schemas for configuration, documents, retrieval results, model output, and API responses. |
| ChromaDB | Persistent local vector database using cosine similarity for semantic search. |
| OpenAI-compatible APIs | Pluggable embedding and chat-completion providers; the endpoint can be hosted or local. |
| pypdf | Text extraction from PDF pages while preserving page metadata for citations. |
| Streamlit | Small interactive frontend that demonstrates the product flow without duplicating backend logic. |
| Docker Compose | Reproducible multi-container development, persistent storage, health checks, and one-shot evaluation jobs. |
| pytest, Ruff, Mypy | Automated behavior tests, linting, and static type checking. |

### RAG flow

1. Ingestion extracts pages, splits text into overlapping chunks, and assigns stable chunk IDs.
2. The embedding provider turns chunks and questions into vectors.
3. Chroma returns the most similar chunks, retaining source filename, page, score, and text.
4. The generation prompt labels retrieved passages as untrusted evidence and requires structured JSON claims with quotations.
5. Pydantic validation checks the response, citation IDs, metadata, and quote membership. Only validated claims are rendered in the UI.

## Configure

Copy the example environment file and set provider credentials and models:

```bash
cp .env.example .env
```

The API uses `EMBEDDING_*` and `LLM_*` settings. `OPENAI_API_KEY` may be used as a fallback.

Put ingestion-ready `.pdf` or `.txt` files under `sources/`. By default, the API indexes them at startup and persists the index in the `chroma-data` Docker volume.

## Run the application with Docker

```bash
docker compose up --build
```

Open the frontend at <http://localhost:8501>. The API is available at <http://localhost:8000/docs>. Stop the services with `Ctrl+C`.

## Run the evaluation with Docker

Start the application first, then run the one-shot evaluation container:

```bash
docker compose --profile evaluation run --rm evaluation
```

The evaluator sends every case in `evaluation/dataset.json` to the API and writes `evaluation/results/results.json` and `evaluation/results/results.csv` on the host. Override options after the service name:

```bash
docker compose --profile evaluation run --rm evaluation \
  --top-k 10 \
  --output-dir evaluation/results/latest
```

Reports include retrieval hit rate, document hit rate, reciprocal rank, citation groundedness, refusal accuracy, answerable coverage, false-refusal rate, unsafe-answer rate, and latency. Groundedness checks that citations and quotes trace to retrieved chunks; it does not establish semantic correctness.

## Design notes

If retrieval finds no usable evidence, the system returns “insufficient evidence” instead of guessing. Provider errors remain visible as operational failures instead of being mislabeled as insufficient evidence. PDF layout artifacts such as line breaks and repeated whitespace are handled during quote validation, while the original source excerpt is retained for traceability.

The evaluation is deliberately focused on measurable system behavior: retrieval ranking, source-document/page recall, citation traceability, refusal classification, and latency. It does not claim that lexical or citation checks prove semantic correctness.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/start.py
```

This starts both processes locally. To run them separately, use Uvicorn on port 8000 and Streamlit on port 8501, with `API_BASE_URL` pointing to the API.

## Checks

```bash
python -m pytest
ruff check src tests scripts streamlit_app.py
mypy
```
