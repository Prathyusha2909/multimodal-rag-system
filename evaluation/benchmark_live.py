from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_BASE_URL = "https://prathyusha2909-multimodal-rag-api.onrender.com"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "samples" / "outputs" / "live-latency-benchmark.json"
QUESTIONS = (
    "What is the revenue trend shown in Figure 3?",
    "Compare online and store profit in Table 2.",
    "What is the principal operational risk in the scanned note?",
    "How did renewable electricity adoption change by 2025?",
    "Explain the relationship between sales and profit shown in the graph.",
)


def query(base_url: str, question: str) -> dict:
    payload = json.dumps({"question": question, "top_k": 4}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/v1/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.load(response)
    wall_ms = round((time.perf_counter() - started) * 1000)
    return {
        "question": question,
        "retrieval_ms": result["retrieval_ms"],
        "generation_ms": result["generation_ms"],
        "wall_ms": wall_ms,
        "citations": len(result["citations"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the deployed multimodal RAG API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = [query(args.base_url, question) for question in QUESTIONS]
    report = {
        "measured_at": datetime.now(UTC).isoformat(),
        "environment": "Render Free, deterministic local answer generator, top_k=4",
        "base_url": args.base_url,
        "queries": rows,
        "summary": {
            "query_count": len(rows),
            "mean_retrieval_ms": round(statistics.mean(row["retrieval_ms"] for row in rows)),
            "mean_wall_ms": round(statistics.mean(row["wall_ms"] for row in rows)),
            "warm_mean_retrieval_ms": round(statistics.mean(row["retrieval_ms"] for row in rows[1:])),
            "first_query_retrieval_ms": rows[0]["retrieval_ms"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
