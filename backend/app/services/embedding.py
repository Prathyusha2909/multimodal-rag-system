from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np

os.environ.setdefault("USE_TF", "0")


class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerProvider:
    """Lazy SentenceTransformer embeddings with a content-addressed vector cache."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Path | str = Path("data/cache"),
        dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        self.dimension = dimension
        self.cache_dir = Path(cache_dir)
        self.model_cache_dir = self.cache_dir / "models" / "sentence-transformers"
        model_key = hashlib.sha256(self.name.encode("utf-8")).hexdigest()[:12]
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
            generated = self._get_model().encode(
                missing_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for index, text, vector in zip(missing_indices, missing_texts, generated):
                embedding = np.asarray(vector, dtype="float32")
                vectors[index] = embedding
                self._save_cached(text, "document", embedding)

        return np.vstack(vectors).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        cached = self._load_cached(text, "query")
        if cached is not None:
            return cached
        vector = self._get_model().encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embedding = np.asarray(vector, dtype="float32")
        self._save_cached(text, "query", embedding)
        return embedding

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.model_cache_dir),
                local_files_only=self._has_cached_model(),
            )
            actual_dimension = self._model.get_sentence_embedding_dimension()
            if actual_dimension != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: configured {self.dimension}, model returns "
                    f"{actual_dimension}"
                )
        return self._model

    def _has_cached_model(self) -> bool:
        model_dir = self.model_cache_dir / f"models--{self.model_name.replace('/', '--')}"
        snapshots = model_dir / "snapshots"
        return snapshots.exists() and any(snapshots.iterdir())

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
