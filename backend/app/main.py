from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.schemas import DocumentSummary, QueryRequest, QueryResponse, StatsResponse
from app.services.cache import IngestionCache
from app.services.chunking import TokenChunker
from app.services.embedding import EmbeddingProvider, SentenceTransformerProvider
from app.services.generator import AnswerGenerator
from app.services.ingestion import DocumentIngestor
from app.services.registry import DocumentRegistry
from app.services.reranker import CrossEncoderReranker, Reranker
from app.services.retriever import HybridRetriever
from app.services.vector_store import FaissVectorStore
from app.services.vision import GeminiVisionAnalyzer

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def create_app(
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    chunker: TokenChunker | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    for directory in (settings.upload_dir, settings.cache_dir, settings.index_dir):
        directory.mkdir(parents=True, exist_ok=True)

    embedder = embedder or SentenceTransformerProvider(
        model_name=settings.embedding_model,
        cache_dir=settings.cache_dir,
    )
    vector_store = FaissVectorStore(embedder, settings.index_dir)
    reranker = reranker or CrossEncoderReranker(settings.reranker_model, settings.cache_dir)
    registry = DocumentRegistry(vector_store)
    retriever = HybridRetriever(vector_store, reranker, settings.retrieval_candidates)
    generator = AnswerGenerator(settings.gemini_api_key, settings.gemini_model)
    ingestor = DocumentIngestor(
        chunker or TokenChunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens),
        IngestionCache(settings.cache_dir),
        GeminiVisionAnalyzer(settings.gemini_api_key, settings.vision_model),
    )
    initialization_lock = threading.Lock()
    initialized = False

    def ensure_initialized() -> None:
        nonlocal initialized
        if initialized:
            return
        with initialization_lock:
            if not initialized:
                registry.initialize_demo()
                initialized = True

    app = FastAPI(
        title=settings.app_name,
        description="Multimodal document intelligence with hybrid retrieval and citations.",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_origin_regex=settings.frontend_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "service": settings.app_name}

    @app.get("/api/v1/stats", response_model=StatsResponse)
    def stats() -> dict:
        ensure_initialized()
        return registry.stats(vector_store.name)

    @app.get("/api/v1/documents", response_model=list[DocumentSummary])
    def documents() -> list[dict]:
        ensure_initialized()
        return registry.documents()

    @app.post("/api/v1/documents/upload", response_model=DocumentSummary, status_code=201)
    async def upload_document(file: UploadFile = File(...)) -> dict:
        ensure_initialized()
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

        (settings.upload_dir / filename).write_bytes(content)
        registry.add(chunks)
        return next(doc for doc in registry.documents() if doc["id"] == chunks[0].document_id)

    @app.post("/api/v1/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> dict:
        ensure_initialized()
        retrieval_started = time.perf_counter()
        filters = set(request.document_ids) if request.document_ids else None
        hits = retriever.search(request.question, request.top_k, filters)
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

        generation_started = time.perf_counter()
        answer, model = generator.generate(request.question, hits)
        generation_ms = int((time.perf_counter() - generation_started) * 1000)
        return {
            "answer": answer,
            "citations": [
                {
                    "index": index,
                    "document_name": hit.chunk.document_name,
                    "page": hit.chunk.page,
                    "modality": hit.chunk.modality,
                    "excerpt": hit.chunk.content,
                    "score": round(hit.score, 4),
                }
                for index, hit in enumerate(hits, start=1)
            ],
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "model": model,
        }

    @app.post("/api/v1/demo/reset")
    def reset_demo() -> dict[str, int | str]:
        nonlocal initialized
        registry.reset_demo()
        initialized = True
        return {"status": "reset", "chunks": len(vector_store.chunks)}

    return app


app = create_app()
