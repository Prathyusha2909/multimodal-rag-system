# Demo

`demo.mp4` is generated from the repository screenshots by `scripts/generate_demo.py`. It walks through document upload, a multimodal query, and inspectable citation evidence.

The demonstrated retrieval path uses `SentenceTransformer("BAAI/bge-small-en-v1.5")`, persistent FAISS vector search, BM25 candidate fusion, and SentenceTransformers `CrossEncoder` reranking.

For a live demo, start the API and frontend, then open `http://localhost:5173`.
