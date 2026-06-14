from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multimodal RAG Studio"
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_origin: str = "http://localhost:5173"
    frontend_origin_regex: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    vision_model: str = "gemini-2.5-flash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_runtime: str = "sentence-transformers"
    embedding_batch_size: int = 4
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_runtime: str = "sentence-transformers"
    fastembed_reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    fastembed_threads: int = 1
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    retrieval_candidates: int = 10
    upload_dir: Path = Path(__file__).resolve().parents[1] / "data" / "uploads"
    cache_dir: Path = Path(__file__).resolve().parents[1] / "data" / "cache"
    index_dir: Path = Path(__file__).resolve().parents[1] / "data" / "index"

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return settings
