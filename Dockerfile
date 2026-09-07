FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY streamlit_app.py ./streamlit_app.py
COPY sources ./sources
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir ".[dev]"

EXPOSE 8000 8501
ENV API_BASE_URL=http://127.0.0.1:8000 \
    CHROMA_PATH=/app/data/chroma \
    SOURCES_DIR=/app/sources \
    AUTO_INDEX_SOURCES=true
CMD ["python", "scripts/start.py"]
