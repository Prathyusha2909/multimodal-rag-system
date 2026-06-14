from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np

from app.domain import DocumentChunk
from app.services.embedding import EmbeddingProvider


class FaissVectorStore:
    """Persistent cosine-similarity FAISS index plus JSON chunk metadata."""

    def __init__(self, embedder: EmbeddingProvider, index_dir: Path | str) -> None:
        self.embedder = embedder
        self.name = f"faiss:{embedder.name}"
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "vectors.faiss"
        self.metadata_path = self.index_dir / "chunks.json"
        self.manifest_path = self.index_dir / "manifest.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: list[DocumentChunk] = []
        self._index = faiss.IndexFlatIP(self.embedder.dimension)
        self._load()

    def add(self, chunks: Iterable[DocumentChunk]) -> None:
        existing = {chunk.id for chunk in self._chunks}
        new_chunks = [chunk for chunk in chunks if chunk.id not in existing]
        if not new_chunks:
            return
        vectors = self.embedder.embed_documents(chunk.content for chunk in new_chunks)
        self._index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self._chunks.extend(new_chunks)
        self._persist()

    def clear(self) -> None:
        self._chunks.clear()
        self._index = faiss.IndexFlatIP(self.embedder.dimension)
        self._persist()

    def search(
        self,
        query: str,
        limit: int = 10,
        document_ids: set[str] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        if not self._chunks:
            return []
        query_vector = np.ascontiguousarray([self.embedder.embed_query(query)], dtype="float32")
        search_limit = len(self._chunks) if document_ids else min(limit, len(self._chunks))
        scores, indices = self._index.search(query_vector, search_limit)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self._chunks[index]
            if document_ids and chunk.document_id not in document_ids:
                continue
            results.append((chunk, float(score)))
            if len(results) >= limit:
                break
        return results

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        return tuple(self._chunks)

    def _load(self) -> None:
        if not (self.index_path.exists() and self.metadata_path.exists() and self.manifest_path.exists()):
            return
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest != {"model": self.embedder.name, "dimension": self.embedder.dimension}:
                return
            chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(self.index_path))
            if index.d != self.embedder.dimension or index.ntotal != len(chunks):
                return
            self._chunks = [DocumentChunk(**chunk) for chunk in chunks]
            self._index = index
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self._chunks = []
            self._index = faiss.IndexFlatIP(self.embedder.dimension)

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps([chunk.to_dict() for chunk in self._chunks], indent=2),
            encoding="utf-8",
        )
        self.manifest_path.write_text(
            json.dumps({"model": self.embedder.name, "dimension": self.embedder.dimension}, indent=2),
            encoding="utf-8",
        )
