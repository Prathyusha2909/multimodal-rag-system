from __future__ import annotations

import re
from typing import Protocol

from app.domain import DocumentChunk, Modality


class TokenEncoding(Protocol):
    def encode(self, text: str): ...

    def decode(self, tokens) -> str: ...


class RegexTokenEncoding:
    """Network-free tokenizer used for tests and offline fallback."""

    def encode(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def decode(self, tokens: list[str]) -> str:
        value = " ".join(tokens)
        return re.sub(r"\s+([,.;:!?%)\]])", r"\1", value)


class TokenChunker:
    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 100,
        encoding: TokenEncoding | None = None,
    ) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Chunk size must be positive and overlap must be smaller than it")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._encoding = encoding

    @property
    def encoding(self) -> TokenEncoding:
        if self._encoding is None:
            try:
                import tiktoken

                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = RegexTokenEncoding()
        return self._encoding

    def split(
        self,
        document_id: str,
        filename: str,
        page: int,
        text: str,
        modality: Modality = "text",
        title: str | None = None,
        id_prefix: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> list[DocumentChunk]:
        tokens = self.encoding.encode(text.strip())
        if not tokens:
            return []
        step = self.chunk_size - self.overlap
        chunks = []
        prefix = id_prefix or f"{document_id}-{page}-{modality}"
        for chunk_index, start in enumerate(range(0, len(tokens), step), start=1):
            end = min(start + self.chunk_size, len(tokens))
            content = self.encoding.decode(tokens[start:end]).strip()
            if not content:
                continue
            chunk_metadata = {
                "filename": filename,
                "page": page,
                "chunk_index": chunk_index,
                "token_start": start,
                "token_end": end,
                "token_count": end - start,
                **(metadata or {}),
            }
            chunks.append(
                DocumentChunk(
                    id=f"{prefix}-{chunk_index}",
                    document_id=document_id,
                    document_name=filename,
                    page=page,
                    modality=modality,
                    title=title or f"Page {page}",
                    content=content,
                    metadata=chunk_metadata,
                )
            )
            if end == len(tokens):
                break
        return chunks
