"""Index local documents optionally, then print semantic search passages."""

import argparse
import os
import sys
from pathlib import Path

from regulatory_rag.ingestion import ingest_document
from regulatory_rag.models import ChunkingConfig
from regulatory_rag.providers import OpenAICompatibleEmbeddings
from regulatory_rag.retrieval import Retriever
from regulatory_rag.store import ChromaStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, action="append", default=[], metavar="FILE")
    parser.add_argument("--db", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", default="regulatory_documents")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        if not args.question.strip() or args.top_k <= 0:
            raise ValueError("Provide a nonblank question and a positive --top-k")
        config = ChunkingConfig(chunk_size=args.chunk_size, overlap=args.overlap)
        provider = OpenAICompatibleEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )
        store = ChromaStore(
            args.db, embedding_id=provider.embedding_id, collection_name=args.collection
        )
        retriever = Retriever(provider, store)
        for path in args.index:
            count = retriever.index(ingest_document(path, config))
            print(f"Indexed {count} chunks from {path.name}.")
        results = retriever.search(args.question, top_k=args.top_k)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not results:
        print("No passages indexed. Add a document with --index path/to/document.pdf.")
    for rank, passage in enumerate(results, start=1):
        page = str(passage.page_number) if passage.page_number is not None else "n/a"
        print(
            f"\n[{rank}] score={passage.score:.4f} source={passage.source_document} "
            f"page={page}\nchunk_id={passage.chunk_id}\n{passage.text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
