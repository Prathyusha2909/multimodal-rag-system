from __future__ import annotations

from collections import Counter, defaultdict

from app.domain import DocumentChunk
from app.services.sample_corpus import build_sample_corpus
from app.services.vector_store import FaissVectorStore


class DocumentRegistry:
    def __init__(self, vector_store: FaissVectorStore) -> None:
        self.vector_store = vector_store
        self._chunks: list[DocumentChunk] = list(vector_store.chunks)

    def initialize_demo(self) -> None:
        existing = {chunk.id for chunk in self._chunks}
        missing = [chunk for chunk in build_sample_corpus() if chunk.id not in existing]
        if missing:
            self.add(missing)

    def reset_demo(self) -> None:
        self.vector_store.clear()
        self._chunks = build_sample_corpus()
        self.vector_store.add(self._chunks)

    def add(self, chunks: list[DocumentChunk]) -> None:
        existing = {chunk.id for chunk in self._chunks}
        new_chunks = [chunk for chunk in chunks if chunk.id not in existing]
        self._chunks.extend(new_chunks)
        self.vector_store.add(new_chunks)

    def documents(self) -> list[dict]:
        grouped: dict[str, list[DocumentChunk]] = defaultdict(list)
        for chunk in self._chunks:
            grouped[chunk.document_id].append(chunk)
        return [
            {
                "id": document_id,
                "name": chunks[0].document_name,
                "pages": max(chunk.page for chunk in chunks),
                "status": "ready",
                "modalities": sorted({chunk.modality for chunk in chunks}),
                "chunks": len(chunks),
            }
            for document_id, chunks in grouped.items()
        ]

    def stats(self, index_backend: str) -> dict:
        modalities = Counter(chunk.modality for chunk in self._chunks)
        return {
            "documents": len({chunk.document_id for chunk in self._chunks}),
            "pages": len({(chunk.document_id, chunk.page) for chunk in self._chunks}),
            "chunks": len(self._chunks),
            "modalities": dict(modalities),
            "index_backend": index_backend,
        }
