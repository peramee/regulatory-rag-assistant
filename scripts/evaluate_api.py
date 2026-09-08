"""Run the evaluation suite against the containerized HTTP API."""

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx


class APIEvaluationService:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def answer_question_timed(self, question: str, *, top_k: int) -> Any:
        from regulatory_rag.models import RAGResponse
        from regulatory_rag.service import AnswerExecution

        started = perf_counter()
        response = httpx.post(
            f"{self.base_url}/query",
            json={"question": question, "top_k": top_k},
            timeout=httpx.Timeout(self.timeout, connect=5.0),
        )
        response.raise_for_status()
        result = RAGResponse.model_validate_json(response.content)
        total = perf_counter() - started
        retrieval = float(response.headers.get("X-Retrieval-Latency-Ms", "0")) / 1000
        return AnswerExecution(result, retrieval_seconds=retrieval, total_seconds=total)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from regulatory_rag.evaluation import (
        console_summary,
        evaluate,
        load_dataset,
        write_report,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--api-url", default=os.environ.get("EVALUATION_API_URL", "http://api:8000")
    )
    parser.add_argument(
        "--timeout", type=float, default=float(os.environ.get("API_TIMEOUT_SECONDS", "120"))
    )
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        report = evaluate(
            APIEvaluationService(args.api_url, args.timeout),  # type: ignore[arg-type]
            dataset,
            top_k=args.top_k,
            dataset_path=str(args.dataset),
        )
        json_path, csv_path = write_report(report, args.output_dir)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(console_summary(report))
    print(f"JSON results: {json_path}")
    print(f"CSV results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
