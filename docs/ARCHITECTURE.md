# Architecture

## Scope

This is a local portfolio prototype. The implemented path is intentionally small enough to inspect and explain end to end.

## Ingestion

`DocumentIngestor` accepts PDF, common image formats, text, Markdown, and CSV.

- `pypdf` extracts available text from each PDF page.
- Tesseract extracts text from image uploads when its local binary is installed.
- Pages without extractable PDF text are marked as scans rather than silently treated as understood.
- The bundled demo corpus contains curated modality labels and visual descriptions corresponding to the synthetic PDFs in `samples/documents/`.

Every chunk stores document ID, filename, page, modality, content, and optional figure/table metadata.

## Retrieval

The prototype uses two local retrieval signals:

1. Deterministic hash vectors provide lightweight semantic token matching.
2. A BM25 implementation preserves exact labels, values, and technical terms.

Candidates are fused and reranked using semantic score, lexical score, token overlap, and explicit modality terms such as `chart`, `figure`, and `table`.

## Generation

The default synthesizer extracts relevant sentences and appends citation markers. When `GEMINI_API_KEY` is configured, the same retrieved text context can be sent to Gemini. Citations are constructed from retrieval metadata rather than invented by the generator.

## Storage

The current demo index is in memory and resets when the API restarts. Uploaded source files are stored in `backend/data/uploads/`. Persistence and background workers are intentionally outside the current scope.

## Evaluation

`evaluation/run_deepeval.py` creates DeepEval `LLMTestCase` objects and measures:

- Retrieval Hit@4
- expected cited-page coverage
- required-fact coverage

The metrics are deterministic and operate on the published synthetic test set. They do not estimate broad answer quality or hallucination probability.

