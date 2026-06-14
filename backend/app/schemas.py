from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    document_ids: list[str] | None = None


class Citation(BaseModel):
    index: int
    document_name: str
    page: int
    modality: Literal["text", "table", "chart", "image", "scan"]
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval_ms: int
    generation_ms: int
    model: str


class DocumentSummary(BaseModel):
    id: str
    name: str
    pages: int
    status: str
    modalities: list[str]
    chunks: int


class StatsResponse(BaseModel):
    documents: int
    pages: int
    chunks: int
    modalities: dict[str, int]
    index_backend: str
