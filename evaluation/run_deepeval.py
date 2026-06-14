from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.embedding import FastEmbedProvider  # noqa: E402
from app.services.generator import AnswerGenerator  # noqa: E402
from app.services.registry import DocumentRegistry  # noqa: E402
from app.services.reranker import CrossEncoderReranker  # noqa: E402
from app.services.retriever import HybridRetriever  # noqa: E402
from app.services.vector_store import FaissVectorStore  # noqa: E402


class MetadataMetric(BaseMetric):
    threshold = 1.0
    async_mode = False

    async def a_measure(self, test_case, *args, **kwargs):
        return await asyncio.to_thread(self.measure, test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)


class RetrievalHitAtK(MetadataMetric):
    @property
    def __name__(self):
        return "Retrieval Hit@4"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected = set(test_case.metadata["expected_chunk_ids"])
        retrieved = set(test_case.metadata["retrieved_chunk_ids"])
        self.score = 1.0 if expected & retrieved else 0.0
        self.success = self.score >= self.threshold
        self.reason = "Expected evidence was retrieved." if self.success else "Expected evidence was missed."
        return self.score


class CitedPageCoverage(MetadataMetric):
    @property
    def __name__(self):
        return "Cited Page Coverage"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected = set(test_case.metadata["expected_pages"])
        retrieved = set(test_case.metadata["retrieved_pages"])
        self.score = len(expected & retrieved) / len(expected) if expected else 1.0
        self.success = self.score >= self.threshold
        self.reason = f"Covered {len(expected & retrieved)} of {len(expected)} expected pages."
        return self.score


class RequiredFactCoverage(MetadataMetric):
    @property
    def __name__(self):
        return "Required Fact Coverage"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        facts = test_case.metadata["required_facts"]
        output = (test_case.actual_output or "").lower()
        matched = [fact for fact in facts if fact.lower() in output]
        self.score = len(matched) / len(facts) if facts else 1.0
        self.success = self.score >= self.threshold
        self.reason = f"Included {len(matched)} of {len(facts)} required facts."
        return self.score


def main() -> None:
    cases = json.loads((ROOT / "evaluation" / "test_set.json").read_text(encoding="utf-8"))
    cache_dir = ROOT / "backend" / "data" / "cache"
    index_dir = ROOT / "backend" / "data" / "index" / "evaluation"
    embedder = FastEmbedProvider("BAAI/bge-small-en-v1.5", cache_dir)
    store = FaissVectorStore(embedder, index_dir)
    registry = DocumentRegistry(store)
    registry.reset_demo()
    reranker = CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2", cache_dir)
    retriever = HybridRetriever(store, reranker, candidate_limit=10)
    generator = AnswerGenerator()
    metric_types = [RetrievalHitAtK, CitedPageCoverage, RequiredFactCoverage]

    rows = []
    totals = {metric_type().__name__: 0.0 for metric_type in metric_types}
    for case in cases:
        hits = retriever.search(case["question"], limit=4)
        answer, model = generator.generate(case["question"], hits)
        test_case = LLMTestCase(
            name=case["id"],
            input=case["question"],
            actual_output=answer,
            retrieval_context=[hit.chunk.content for hit in hits],
            metadata={
                **case,
                "retrieved_chunk_ids": [hit.chunk.id for hit in hits],
                "retrieved_pages": [hit.chunk.page for hit in hits],
            },
        )
        scores = {}
        for metric_type in metric_types:
            metric = metric_type()
            score = metric.measure(test_case)
            scores[metric.__name__] = round(score, 4)
            totals[metric.__name__] += score
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer": answer,
                "model": model,
                "retrieved_chunk_ids": test_case.metadata["retrieved_chunk_ids"],
                "scores": scores,
            }
        )

    summary = {
        name: round(total / len(cases), 4)
        for name, total in totals.items()
    }
    output = {
        "framework": "DeepEval 3.9.9 custom metrics over BGE-small, FAISS, BM25, and MiniLM reranking",
        "test_cases": len(cases),
        "summary": summary,
        "results": rows,
    }
    output_path = ROOT / "samples" / "outputs" / "deepeval-results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    report = [
        "# Evaluation Results",
        "",
        "DeepEval 3.9.9 was used with deterministic custom metrics over the real BGE-small, FAISS, BM25, and MiniLM reranking pipeline; no external judge model was used.",
        "",
        f"- Test cases: {len(cases)}",
        *[f"- {name}: {score:.1%}" for name, score in summary.items()],
        "",
        "Full per-question results are in `deepeval-results.json`.",
    ]
    (output_path.parent / "evaluation-summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
