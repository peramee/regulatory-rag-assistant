"""Index an optional corpus, evaluate the RAG service, then write JSON and CSV results."""

import argparse
import os
import sys
from pathlib import Path


def corpus_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.pdf") if "originals" not in path.parts)


def main(argv: list[str] | None = None) -> int:
    # Make `python scripts/evaluate.py` work from a source checkout as documented.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from regulatory_rag.evaluation import console_summary, evaluate, load_dataset, write_report
    from regulatory_rag.ingestion import ingest_document
    from regulatory_rag.providers import OpenAICompatibleEmbeddings, OpenAICompatibleLLM
    from regulatory_rag.retrieval import Retriever
    from regulatory_rag.service import RAGService
    from regulatory_rag.store import ChromaStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--db", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", default="transaction_reporting_evaluation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", type=Path, action="append", default=[], metavar="FILE")
    parser.add_argument("--index-corpus", action="store_true")
    parser.add_argument("--corpus-dir", type=Path, default=Path("sources/transaction_reporting"))
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        embeddings = OpenAICompatibleEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )
        store = ChromaStore(
            args.db, embedding_id=embeddings.embedding_id, collection_name=args.collection
        )
        service = RAGService(Retriever(embeddings, store), OpenAICompatibleLLM())
        files = args.index + (corpus_files(args.corpus_dir) if args.index_corpus else [])
        if args.index_corpus and not files:
            raise ValueError(
                f"No prepared PDFs found in {args.corpus_dir}; "
                "download the local source corpus first"
            )
        for path in files:
            count = service.retriever.index(ingest_document(path))
            print(f"Indexed {count} chunks from {path}.")
        report = evaluate(service, dataset, top_k=args.top_k, dataset_path=str(args.dataset))
        json_path, csv_path = write_report(report, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(console_summary(report))
    print(f"JSON results: {json_path}")
    print(f"CSV results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
