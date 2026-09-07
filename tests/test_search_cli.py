import runpy
from pathlib import Path

import pytest

from regulatory_rag import providers


def test_demo_indexes_and_prints_passages(tmp_path: Path, monkeypatch, capsys) -> None:
    class FakeEmbeddings:
        embedding_id = "cli-test"

        def __init__(self, **kwargs):
            pass

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(providers, "OpenAICompatibleEmbeddings", FakeEmbeddings)
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/search.py"))["main"]
    path = tmp_path / "regulation.txt"
    path.write_text("Submit annual reports.", encoding="utf-8")
    db = str(tmp_path / "db")
    assert main(["What reporting obligations apply?", "--index", str(path), "--db", db]) == 0
    output = capsys.readouterr().out
    assert "Indexed 1 chunks" in output
    assert "score=1.0000 source=regulation.txt page=n/a" in output
    assert "chunk_id=" in output
    assert "Submit annual reports." in output
    assert main(["Reporting?", "--db", db]) == 0
    assert "Submit annual reports." in capsys.readouterr().out


def test_demo_missing_credentials_reports_actionable_error(monkeypatch, capsys) -> None:
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/search.py"))["main"]
    assert main(["reporting"]) == 1
    assert "EMBEDDING_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [[" "], ["reporting", "--top-k", "0"]])
def test_demo_invalid_question_or_top_k(arguments, capsys) -> None:
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/search.py"))["main"]
    assert main(arguments) == 1
    assert "Error:" in capsys.readouterr().err
