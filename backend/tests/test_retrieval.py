import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from app.domain import DocumentChunk
from app.services.registry import DocumentRegistry
from app.services.retriever import HybridRetriever
from app.services.vector_store import FaissVectorStore
from tests.fakes import FakeEmbeddingProvider, fake_reranker


class FaissVectorStoreTests(unittest.TestCase):
    def test_index_persists_chunks_and_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            first = FaissVectorStore(FakeEmbeddingProvider(), directory)
            registry = DocumentRegistry(first)
            registry.initialize_demo()
            first_result = first.search("revenue chart", 1)[0][0]

            second = FaissVectorStore(FakeEmbeddingProvider(), directory)
            second_result = second.search("revenue chart", 1)[0][0]

            self.assertEqual(len(second.chunks), len(first.chunks))
            self.assertTrue((Path(directory) / "vectors.faiss").exists())
            self.assertEqual(second_result.id, first_result.id)


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = FaissVectorStore(FakeEmbeddingProvider(), self.temp_dir.name)
        self.registry = DocumentRegistry(self.store)
        self.registry.initialize_demo()
        self.retriever = HybridRetriever(self.store, fake_reranker(), candidate_limit=10)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_chart_query_prioritizes_revenue_figure(self):
        hits = self.retriever.search("What revenue trend is shown in the chart?", limit=3)

        self.assertEqual(hits[0].chunk.id, "nova-08-chart")
        self.assertEqual(hits[0].chunk.modality, "chart")

    def test_table_query_retrieves_segment_performance(self):
        hits = self.retriever.search("Compare the online and stores profit in Table 2", limit=2)

        self.assertEqual(hits[0].chunk.id, "nova-09-table")

    def test_document_filter_excludes_other_reports(self):
        hits = self.retriever.search("renewable energy", limit=4, document_ids={"nova-2025"})

        self.assertTrue(hits)
        self.assertTrue(all(hit.chunk.document_id == "nova-2025" for hit in hits))

    def test_reranker_reduces_candidates_to_requested_limit(self):
        hits = self.retriever.search("revenue profit chart table", limit=3)

        self.assertEqual(len(hits), 3)
        self.assertTrue(all(hit.rerank_score for hit in hits))

    def test_summary_query_prefers_diverse_substantive_sections(self):
        self.retriever.reranker.rerank = Mock(
            side_effect=AssertionError("summary retrieval should not load the cross-encoder")
        )
        self.registry.add(
            [
                DocumentChunk(
                    id="job-1",
                    document_id="job-description",
                    document_name="Data_Scientist.pdf",
                    page=1,
                    modality="text",
                    title="Equal opportunity statement",
                    content=(
                        "We are an equal opportunity employer. We promote all talents regardless "
                        "of age, disability, gender identity, religion, or any characteristic "
                        "that could be subject to discrimination."
                    ),
                ),
                DocumentChunk(
                    id="job-2",
                    document_id="job-description",
                    document_name="Data_Scientist.pdf",
                    page=2,
                    modality="text",
                    title="Role overview",
                    content=(
                        "The Associate Data Scientist builds machine-learning solutions, analyzes "
                        "business data, and communicates findings to stakeholders."
                    ),
                ),
                DocumentChunk(
                    id="job-3",
                    document_id="job-description",
                    document_name="Data_Scientist.pdf",
                    page=3,
                    modality="text",
                    title="Skills and qualifications",
                    content=(
                        "Required skills include Python, SQL, statistics, data visualization, and "
                        "experience with machine-learning workflows."
                    ),
                ),
                DocumentChunk(
                    id="job-4",
                    document_id="job-description",
                    document_name="Data_Scientist.pdf",
                    page=4,
                    modality="text",
                    title="Responsibilities",
                    content=(
                        "Responsibilities include preparing datasets, evaluating models, documenting "
                        "results, and collaborating with engineering and product teams."
                    ),
                ),
            ]
        )

        hits = self.retriever.search(
            "summarize",
            limit=4,
            document_ids={"job-description"},
        )

        self.assertGreaterEqual(len(hits), 3)
        self.assertNotIn("job-1", {hit.chunk.id for hit in hits})
        self.assertGreaterEqual(len({hit.chunk.page for hit in hits}), 3)


if __name__ == "__main__":
    unittest.main()
