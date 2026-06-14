import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.embedding import FastEmbedProvider, SentenceTransformerProvider
from app.services.registry import DocumentRegistry
from app.services.reranker import CrossEncoderReranker
from app.services.vector_store import FaissVectorStore


CACHE_ROOT = ROOT / "data" / "cache"
INDEX_DIR = ROOT / "data" / "index"


def main() -> None:
    runtime = os.getenv("EMBEDDING_RUNTIME", "sentence-transformers")
    threads = int(os.getenv("FASTEMBED_THREADS", "1"))
    if runtime == "fastembed":
        embedder = FastEmbedProvider(
            "BAAI/bge-small-en-v1.5",
            CACHE_ROOT,
            threads=threads,
        )
        reranker = CrossEncoderReranker(
            "Xenova/ms-marco-MiniLM-L-6-v2",
            CACHE_ROOT,
            runtime="fastembed",
            threads=threads,
        )
    else:
        embedder = SentenceTransformerProvider("BAAI/bge-small-en-v1.5", CACHE_ROOT)
        reranker = CrossEncoderReranker(
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            CACHE_ROOT,
        )

    embedder._get_model()
    reranker._get_model()
    store = FaissVectorStore(embedder, INDEX_DIR)
    DocumentRegistry(store).reset_demo()
    print(f"Cached {runtime} models and built the demo FAISS index.")


if __name__ == "__main__":
    main()
