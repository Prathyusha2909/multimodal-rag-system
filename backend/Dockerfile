FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_TF=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    EMBEDDING_RUNTIME=fastembed \
    RERANKER_RUNTIME=fastembed \
    FASTEMBED_THREADS=1 \
    EMBEDDING_BATCH_SIZE=4

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts

RUN python scripts/download_models.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
