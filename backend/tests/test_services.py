import unittest

from app.services.generator import AnswerGenerator
from app.services.ingestion import DocumentIngestor
from app.services.registry import DocumentRegistry
from app.services.retriever import HybridRetriever
from app.services.vector_store import MemoryVectorStore


class IngestionTests(unittest.TestCase):
    def test_plain_text_is_chunked_and_indexable(self):
        chunks = DocumentIngestor().ingest(
            "notes.txt",
            b"Revenue increased because online conversion improved.",
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].modality, "text")
        self.assertIn("Revenue increased", chunks[0].content)

    def test_unsupported_file_type_is_rejected(self):
        with self.assertRaises(ValueError):
            DocumentIngestor().ingest("archive.zip", b"not a document")


class GenerationTests(unittest.TestCase):
    def test_local_answer_contains_citation_markers(self):
        store = MemoryVectorStore()
        registry = DocumentRegistry(store)
        registry.reset_demo()
        hits = HybridRetriever(store).search("revenue trend", limit=2)

        answer, model = AnswerGenerator().generate("What is the revenue trend?", hits)

        self.assertIn("[1]", answer)
        self.assertEqual(model, "local-grounded-synthesizer")


if __name__ == "__main__":
    unittest.main()

