import unittest

from app.services.embedding import HashEmbedding, cosine_similarity
from app.services.registry import DocumentRegistry
from app.services.retriever import HybridRetriever
from app.services.vector_store import MemoryVectorStore


class EmbeddingTests(unittest.TestCase):
    def test_embedding_is_normalized_and_deterministic(self):
        embedder = HashEmbedding(dimensions=64)
        first = embedder.embed("revenue growth chart")
        second = embedder.embed("revenue growth chart")

        self.assertEqual(first, second)
        self.assertAlmostEqual(cosine_similarity(first, first), 1.0)


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryVectorStore(HashEmbedding(dimensions=128))
        self.registry = DocumentRegistry(self.store)
        self.registry.reset_demo()
        self.retriever = HybridRetriever(self.store)

    def test_chart_query_prioritizes_revenue_figure(self):
        hits = self.retriever.search("What revenue trend is shown in the chart?", limit=3)

        self.assertEqual(hits[0].chunk.id, "nova-08-chart")
        self.assertEqual(hits[0].chunk.modality, "chart")

    def test_table_query_retrieves_segment_performance(self):
        hits = self.retriever.search("Compare the online and stores profit in Table 2", limit=2)

        self.assertEqual(hits[0].chunk.id, "nova-09-table")

    def test_document_filter_excludes_other_reports(self):
        hits = self.retriever.search(
            "renewable energy",
            limit=4,
            document_ids={"nova-2025"},
        )

        self.assertTrue(hits)
        self.assertTrue(all(hit.chunk.document_id == "nova-2025" for hit in hits))


if __name__ == "__main__":
    unittest.main()

