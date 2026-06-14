from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedProvider:
    """Lazy BGE embeddings with a content-addressed on-disk vector cache."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Path | str = Path("data/cache"),
        dimension: int | None = None,
    ) -> None:
        self.name = model_name
        self.dimension = dimension or self._model_dimension(model_name)
        self.cache_dir = Path(cache_dir)
        self.model_cache_dir = self.cache_dir / "models"
        model_key = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
        self.embedding_cache_dir = self.cache_dir / "embeddings" / model_key
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype="float32")

        vectors: list[np.ndarray | None] = [None] * len(values)
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        for index, value in enumerate(values):
            cached = self._load_cached(value, "document")
            if cached is None:
                missing_texts.append(value)
                missing_indices.append(index)
            else:
                vectors[index] = cached

        if missing_texts:
            generated = list(self._get_model().embed(missing_texts))
            for index, text, vector in zip(missing_indices, missing_texts, generated):
                normalized = self._normalize(vector)
                vectors[index] = normalized
                self._save_cached(text, "document", normalized)

        return np.vstack(vectors).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        cached = self._load_cached(text, "query")
        if cached is not None:
            return cached
        vector = next(iter(self._get_model().query_embed(text)))
        normalized = self._normalize(vector)
        self._save_cached(text, "query", normalized)
        return normalized

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.name,
                cache_dir=str(self.model_cache_dir),
                lazy_load=True,
            )
        return self._model

    def _cache_path(self, text: str, kind: str) -> Path:
        digest = hashlib.sha256(f"{kind}\0{text}".encode("utf-8")).hexdigest()
        return self.embedding_cache_dir / f"{digest}.npy"

    def _load_cached(self, text: str, kind: str) -> np.ndarray | None:
        path = self._cache_path(text, kind)
        if not path.exists():
            return None
        vector = np.load(path, allow_pickle=False).astype("float32")
        return vector if vector.shape == (self.dimension,) else None

    def _save_cached(self, text: str, kind: str, vector: np.ndarray) -> None:
        np.save(self._cache_path(text, kind), vector, allow_pickle=False)

    @staticmethod
    def _normalize(vector) -> np.ndarray:
        array = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(array)) or 1.0
        return array / norm

    @staticmethod
    def _model_dimension(model_name: str) -> int:
        from fastembed import TextEmbedding

        for model in TextEmbedding.list_supported_models():
            if model["model"] == model_name:
                return int(model["dim"])
        raise ValueError(f"Unsupported FastEmbed model: {model_name}")
