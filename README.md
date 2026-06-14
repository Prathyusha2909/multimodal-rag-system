# Multimodal RAG System

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-UI-61DAFB)](https://react.dev/)
[![Retrieval](https://img.shields.io/badge/Retrieval-SentenceTransformers%20%2B%20FAISS-8A2BE2)](backend/app/services/embedding.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-b8ee45.svg)](LICENSE)

A portfolio-scale retrieval-augmented generation prototype for PDFs, images, tables, and text files. It uses SentenceTransformer embeddings with FAISS vector search, BM25 hybrid retrieval, cross-encoder reranking, and document/page citations.

![Multimodal RAG dashboard](screenshots/home-page.png)

## Honest Scope

This repository demonstrates a local prototype, not a production deployment.

- Text PDFs are parsed page by page with `pypdf`; detected tables are extracted separately with `pdfplumber`.
- Image uploads use Tesseract OCR when its binary is installed and optional Gemini Vision captioning when an API key is configured.
- The included synthetic reports have curated chart, table, image, and scan descriptions so retrieval can also be tested without external APIs.
- Gemini Vision is applied to uploaded image files. This version does not render every PDF page or embedded PDF figure through a vision model.
- Gemini can be enabled for answer synthesis; the default generator is deterministic and local.
- BGE-small and the MS MARCO cross-encoder are loaded through the `sentence-transformers` package on first use.
- No private documents, internal servers, real hardware, or company data are used.

## Problem Statement

Text-only RAG loses useful context when evidence is presented as a table, chart, scanned note, or image. This prototype keeps text, table, and visual descriptions as modality-aware chunks with source metadata, then retrieves and reranks them together.

## Implemented Architecture

![Architecture](docs/architecture.png)

```text
PDF / image / text / CSV upload
          |
          v
pypdf text + pdfplumber tables + Tesseract/Gemini Vision
          |
          v
500-token chunks + 100-token overlap + source metadata
          |
          +---- SentenceTransformer embeddings -> FAISS vector search
          +---- BM25 lexical search
                         |
                         v
                 top 10 hybrid candidates
                         |
                         v
              SentenceTransformers CrossEncoder
                         |
                         v
          top 3-5 -> local or Gemini answer
                         |
                         v
                inspectable page citations
```

Read the [architecture notes](docs/ARCHITECTURE.md) and [API reference](docs/API.md).

## Real Retrieval Implementation

The application runtime uses model-generated semantic embeddings and a persistent vector index.

```text
SentenceTransformer("BAAI/bge-small-en-v1.5")
        -> normalized 384-dimensional embeddings
        -> FAISS IndexFlatIP vector search
        -> BM25 candidate fusion (top 10)
        -> CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        -> top 3-5 evidence chunks
```

The implementation is visible in [`embedding.py`](backend/app/services/embedding.py), [`vector_store.py`](backend/app/services/vector_store.py), [`retriever.py`](backend/app/services/retriever.py), and [`reranker.py`](backend/app/services/reranker.py).

## Features

- PDF, PNG, JPG, WEBP, TIFF, TXT, Markdown, and CSV ingestion
- Page-level PDF text extraction and separate `pdfplumber` table chunks
- Optional Tesseract OCR and Gemini Vision analysis for image uploads
- 500-token chunks with 100-token overlap and filename, page, chunk, and token metadata
- `BAAI/bge-small-en-v1.5` embeddings through `SentenceTransformer`
- Persistent cosine-similarity FAISS vector search plus BM25 lexical retrieval
- Top-10 candidate retrieval and `cross-encoder/ms-marco-MiniLM-L6-v2` reranking
- On-disk caches for extraction, embeddings, reranker scores, model files, and the vector index
- Deterministic local answer generation
- Optional Gemini 2.5 Flash answer generation and image analysis
- Document, page, modality, excerpt, and reranking score in every citation
- Responsive React interface
- Docker Compose for running the API and UI locally

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend | Python, FastAPI, Pydantic |
| Frontend | React, Vite |
| Retrieval | SentenceTransformers, BGE-small, FAISS, BM25, CrossEncoder |
| Parsing | pypdf, pdfplumber, optional Tesseract OCR |
| Vision | Optional Gemini 2.5 Flash for uploaded images/charts |
| Generation | Local extractive synthesizer, optional Gemini 2.5 Flash |
| Evaluation | DeepEval custom deterministic metrics |
| Packaging | Docker Compose, GitHub Actions |

## Sample Data

The [`samples/`](samples/) directory contains publishable synthetic artifacts:

- two generated PDF reports containing charts, tables, and page text
- a simulated PCIe link-training log
- a simulated GitHub issue
- an example cited API response
- full DeepEval results

The PCIe log and issue simulate validation workflows. They are explicitly marked synthetic and do not claim access to test equipment or internal systems.

## Evaluation

Responses were evaluated using **DeepEval 3.9.9 on a custom eight-question test set**. The runner executes the real SentenceTransformer, FAISS, BM25, and CrossEncoder path, then applies deterministic custom metrics without an external judge model.

| Metric | Result on published sample set |
| --- | ---: |
| Retrieval Hit@4 | 100% |
| Expected cited-page coverage | 100% |
| Required-fact coverage | 95.8% |

These results describe only [`evaluation/test_set.json`](evaluation/test_set.json); they are not claims about production accuracy or general hallucination reduction. See the generated [evaluation summary](samples/outputs/evaluation-summary.md) and [per-question results](samples/outputs/deepeval-results.json).

Reproduce the evaluation:

```bash
python -m pip install -r backend/requirements-eval.txt
python evaluation/run_deepeval.py
```

## Quick Start

### Deploy

[![Deploy API to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Prathyusha2909/multimodal-rag-system)
[![Deploy UI with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FPrathyusha2909%2Fmultimodal-rag-system&project-name=prathyusha2909-multimodal-rag&repository-name=multimodal-rag-system&root-directory=frontend)

Deploy the Render API first, then deploy the Vercel frontend. The checked-in production frontend configuration targets `https://prathyusha2909-multimodal-rag-api.onrender.com`; override `VITE_API_URL` in Vercel if Render assigns a different URL.

The Render Docker image pre-caches both SentenceTransformers models and the demo FAISS index. Render Free spins down after 15 idle minutes and uses an ephemeral filesystem, so uploaded documents disappear after a restart or redeploy. Use a paid Render service with a persistent disk for durable uploads and indexes.

### Docker

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:5173` for the UI or `http://localhost:8000/docs` for OpenAPI.

### Local Development

Generic Windows activation command: `.venv\Scripts\activate`

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn app.main:app --reload
```

Windows Command Prompt activation equivalent: `.venv\Scripts\activate.bat`.

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Set `GEMINI_API_KEY` in `.env` to use Gemini instead of the local synthesizer.

The first API initialization downloads the SentenceTransformers embedding and reranker models. Later runs reuse `backend/data/cache/` and the persistent FAISS index in `backend/data/index/`.

## Usage

1. Upload one of the files from `samples/documents/` or another supported file.
2. Ask a question in the workspace.
3. Inspect the returned document, page, modality, excerpt, and reranking score.

Sample questions:

```text
What is the revenue trend shown in Figure 3?
Compare online and store profit in Table 2.
What is the principal operational risk in the scanned note?
How did renewable electricity adoption change by 2025?
```

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Compare online and store profit in Table 2.","top_k":4}'
```

## Screenshots

| Query | Cited result |
| --- | --- |
| ![Query example](screenshots/query-example.png) | ![Result example](screenshots/result-example.png) |

## Repository Structure

```text
multimodal-rag-system/
|-- backend/              # FastAPI app, tests, caches, and persistent FAISS data
|-- frontend/             # React interface
|-- evaluation/           # DeepEval test set and runner
|-- samples/              # PDFs, logs, issues, and outputs
|-- docs/                 # architecture and API notes
|-- screenshots/          # generated UI screenshots
|-- demo/demo.mp4         # generated walkthrough
|-- scripts/              # sample and visual asset generators
|-- render.yaml           # Render API Blueprint
|-- frontend/vercel.json  # Vercel frontend build configuration
`-- docker-compose.yml
```

## Verification

```bash
cd backend
python -m pytest

cd ../frontend
npm run build
```

## Resume Wording

> Built a multimodal document RAG system that extracts PDF text and tables, analyzes uploaded images with OCR and optional Gemini Vision, indexes 500-token chunks using SentenceTransformer embeddings and FAISS vector search, and reranks hybrid results with a CrossEncoder. Evaluated responses using DeepEval on a custom eight-question test set.

## License

MIT
