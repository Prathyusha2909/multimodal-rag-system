import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.services.embedding import SentenceTransformerProvider
from app.services.registry import DocumentRegistry
from app.services.vector_store import FaissVectorStore


CACHE_ROOT = ROOT / "data" / "cache"
MODEL_CACHE_DIR = CACHE_ROOT / "models" / "sentence-transformers"
INDEX_DIR = ROOT / "data" / "index"


def main() -> None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        cache_folder=str(MODEL_CACHE_DIR),
    )
    CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        cache_dir=str(MODEL_CACHE_DIR),
    )
    store = FaissVectorStore(
        SentenceTransformerProvider("BAAI/bge-small-en-v1.5", CACHE_ROOT),
        INDEX_DIR,
    )
    DocumentRegistry(store).reset_demo()
    print("Cached models and built the demo FAISS index.")


if __name__ == "__main__":
    main()
