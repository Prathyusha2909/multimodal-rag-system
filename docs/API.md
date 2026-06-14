# API Reference

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

## Health

```http
GET /health
```

## Index statistics

```http
GET /api/v1/stats
```

## Documents

```http
GET /api/v1/documents
```

Upload a PDF, image, text file, Markdown file, or CSV:

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@annual-report.pdf"
```

## Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Compare Figure 3 and Table 2.","top_k":4}'
```

The response contains the grounded answer, ranked citations, retrieval timing, generation timing, and model identifier.

Citation `score` is the retriever's internal reranking score. It is not a probability or a calibrated confidence value.
