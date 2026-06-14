from __future__ import annotations

from collections.abc import Iterable

from app.domain import DocumentChunk
from app.services.embedding import HashEmbedding, cosine_similarity


class MemoryVectorStore:
    def __init__(self, embedder: HashEmbedding | None = None) -> None:
        self.embedder = embedder or HashEmbedding()
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: Iterable[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks.append(chunk)
            self._vectors.append(self.embedder.embed(chunk.content))

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()

    def search(
        self,
        query: str,
        limit: int = 10,
        document_ids: set[str] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        query_vector = self.embedder.embed(query)
        results = []
        for chunk, vector in zip(self._chunks, self._vectors):
            if document_ids and chunk.document_id not in document_ids:
                continue
            results.append((chunk, cosine_similarity(query_vector, vector)))
        return sorted(results, key=lambda item: item[1], reverse=True)[:limit]

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        return tuple(self._chunks)
