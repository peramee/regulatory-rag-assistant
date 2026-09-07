"""Start the FastAPI backend and Streamlit frontend together."""

import os
import subprocess
import sys
from pathlib import Path
from time import sleep


def load_dotenv() -> None:
    """Load simple KEY=VALUE pairs without adding a dotenv runtime dependency."""
    path = Path(".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    load_dotenv()
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "regulatory_rag.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            os.environ.get("API_PORT", "8000"),
        ]
    )
    frontend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            os.environ.get("STREAMLIT_PORT", "8501"),
            "--server.headless",
            "true",
        ]
    )
    processes = [api, frontend]
    try:
        while all(process.poll() is None for process in processes):
            sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
    return next((process.returncode or 0 for process in processes if process.returncode), 0)


if __name__ == "__main__":
    raise SystemExit(main())
