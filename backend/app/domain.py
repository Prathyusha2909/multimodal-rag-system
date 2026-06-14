from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Modality = Literal["text", "table", "chart", "image", "scan"]


@dataclass(slots=True)
class DocumentChunk:
    id: str
    document_id: str
    document_name: str
    page: int
    modality: Modality
    content: str
    title: str = ""
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    chunk: DocumentChunk
    semantic_score: float
    lexical_score: float
    rerank_score: float = 0.0

    @property
    def score(self) -> float:
        return self.rerank_score or (0.65 * self.semantic_score + 0.35 * self.lexical_score)

