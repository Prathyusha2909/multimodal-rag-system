from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.schemas import DocumentSummary, QueryRequest, QueryResponse, StatsResponse
from app.services.generator import AnswerGenerator
from app.services.ingestion import DocumentIngestor
from app.services.registry import DocumentRegistry
from app.services.retriever import HybridRetriever
from app.services.vector_store import MemoryVectorStore

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    vector_store = MemoryVectorStore()
    registry = DocumentRegistry(vector_store)
    registry.reset_demo()
    retriever = HybridRetriever(vector_store)
    generator = AnswerGenerator(settings.gemini_api_key, settings.gemini_model)
    ingestor = DocumentIngestor()

    app = FastAPI(
        title=settings.app_name,
        description="Multimodal document intelligence with hybrid retrieval and citations.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "service": settings.app_name}

    @app.get("/api/v1/stats", response_model=StatsResponse)
    def stats() -> dict:
        return registry.stats("local-hash-index")

    @app.get("/api/v1/documents", response_model=list[DocumentSummary])
    def documents() -> list[dict]:
        return registry.documents()

    @app.post("/api/v1/documents/upload", response_model=DocumentSummary, status_code=201)
    async def upload_document(file: UploadFile = File(...)) -> dict:
        filename = Path(file.filename or "upload").name
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds the 25 MB upload limit")
        try:
            chunks = ingestor.ingest(filename, content)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not chunks:
            raise HTTPException(status_code=422, detail="No indexable content was found")

        target = settings.upload_dir / filename
        target.write_bytes(content)
        registry.add(chunks)
        return next(doc for doc in registry.documents() if doc["id"] == chunks[0].document_id)

    @app.post("/api/v1/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> dict:
        retrieval_started = time.perf_counter()
        filters = set(request.document_ids) if request.document_ids else None
        hits = retriever.search(request.question, request.top_k, filters)
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

        generation_started = time.perf_counter()
        answer, model = generator.generate(request.question, hits)
        generation_ms = int((time.perf_counter() - generation_started) * 1000)
        citations = [
            {
                "index": index,
                "document_name": hit.chunk.document_name,
                "page": hit.chunk.page,
                "modality": hit.chunk.modality,
                "excerpt": hit.chunk.content,
                "score": round(hit.score, 4),
            }
            for index, hit in enumerate(hits, start=1)
        ]
        return {
            "answer": answer,
            "citations": citations,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "model": model,
        }

    @app.post("/api/v1/demo/reset")
    def reset_demo() -> dict[str, int | str]:
        registry.reset_demo()
        return {"status": "reset", "chunks": len(vector_store.chunks)}

    return app
app = create_app()
