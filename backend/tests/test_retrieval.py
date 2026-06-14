import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
