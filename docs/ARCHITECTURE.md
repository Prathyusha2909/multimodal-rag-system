# Architecture

## Scope

This is a local portfolio prototype. The implemented path is intentionally small enough to inspect and explain end to end.

## Ingestion

`DocumentIngestor` accepts PDF, common image formats, text, Markdown, and CSV.

- `pypdf` extracts available text from each PDF page.
- `pdfplumber` extracts detected tables into separate table chunks.
- Tesseract extracts text from image uploads when its local binary is installed.
- Gemini Vision optionally produces factual chart, table, or image descriptions for uploaded images.
- Pages without extractable PDF text are marked as scans rather than silently treated as understood.
- The bundled demo corpus contains curated modality labels and visual descriptions corresponding to the synthetic PDFs in `samples/documents/`.

Text is split into 500-token windows with 100-token overlap. Every chunk stores document ID, filename, page, chunk ID, token offsets, modality, content, and optional figure/table metadata.

## Retrieval

The prototype uses two local retrieval signals:

1. `SentenceTransformer("BAAI/bge-small-en-v1.5")` produces normalized 384-dimensional embeddings stored in a persistent FAISS inner-product index.
2. A BM25 implementation preserves exact labels, values, and technical terms.

The retriever forms a pool of ten semantic and lexical candidates. `CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")` scores each query/chunk pair and reranks the pool before the top three to five chunks are sent to generation. A documented heuristic fallback is used only when the local reranker cannot load.

## Generation

The default synthesizer extracts relevant sentences and appends citation markers. When `GEMINI_API_KEY` is configured, the same retrieved text context can be sent to Gemini. Citations are constructed from retrieval metadata rather than invented by the generator.

## Storage And Caching

Uploaded source files are stored in `backend/data/uploads/`. Extracted chunks, SentenceTransformer vectors, model files, and reranker scores are cached under `backend/data/cache/`. FAISS vectors and chunk metadata persist under `backend/data/index/`. These generated directories are excluded from Git.

For Render deployment, the Docker image includes FastEmbed ONNX versions of BGE-small and the MS MARCO MiniLM reranker plus a prebuilt demo index. This keeps live PDF indexing below the free instance memory ceiling while local development and evaluation retain the SentenceTransformers path. The free Render filesystem remains ephemeral at runtime, so uploaded files and runtime index additions are reset when the service restarts. A paid persistent disk is required for durable hosted uploads.

## Multimodal Boundary

Table extraction and uploaded-image understanding are implemented. Gemini Vision can interpret an uploaded chart image, while OCR preserves visible text. This version does not render full PDF pages through Gemini Vision, so chart understanding inside arbitrary uploaded PDFs depends on extractable text unless the chart is uploaded separately as an image.

## Evaluation

`evaluation/run_deepeval.py` executes the real SentenceTransformer, FAISS, BM25, and CrossEncoder path, creates DeepEval `LLMTestCase` objects, and measures:

- Retrieval Hit@4
- expected cited-page coverage
- required-fact coverage

The metrics are deterministic and operate on the published synthetic test set. They do not estimate broad answer quality or hallucination probability.
