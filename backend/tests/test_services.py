import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.services.cache import IngestionCache
from app.services.chunking import RegexTokenEncoding, TokenChunker
from app.services.generator import AnswerGenerator
from app.services.ingestion import DocumentIngestor
from app.services.registry import DocumentRegistry
from app.services.retriever import HybridRetriever
from app.services.vector_store import FaissVectorStore
from tests.fakes import FakeEmbeddingProvider, FakeVisionAnalyzer, NoVisionAnalyzer, fake_reranker

ROOT = Path(__file__).resolve().parents[2]


class ChunkingTests(unittest.TestCase):
    def test_token_chunks_use_500_size_and_100_overlap(self):
        chunker = TokenChunker(500, 100, RegexTokenEncoding())
        text = " ".join(f"token-{index}" for index in range(1200))

        chunks = chunker.split("doc", "notes.txt", 1, text)

        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(chunks[0].metadata["token_count"], 500)
        self.assertEqual(chunks[1].metadata["token_start"], 400)
        self.assertEqual(chunks[0].metadata["filename"], "notes.txt")


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ingestor = DocumentIngestor(
            TokenChunker(500, 100, RegexTokenEncoding()),
            IngestionCache(self.temp_dir.name),
            NoVisionAnalyzer(),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plain_text_is_chunked_and_cached(self):
        content = b"Revenue increased because online conversion improved."
        first = self.ingestor.ingest("notes.txt", content)
        second = self.ingestor.ingest("notes.txt", content)

        self.assertEqual(first[0].to_dict(), second[0].to_dict())
        self.assertIn("Revenue increased", first[0].content)
        self.assertTrue(list((Path(self.temp_dir.name) / "ingestion").glob("*.json")))

    def test_pdf_tables_are_separate_table_chunks(self):
        content = (ROOT / "samples" / "documents" / "Nova_Retail_Annual_Report_2025.pdf").read_bytes()

        chunks = self.ingestor.ingest("Nova.pdf", content)
        tables = [chunk for chunk in chunks if chunk.modality == "table"]

        self.assertTrue(tables)
        self.assertTrue(any(chunk.page == 9 and "128M" in chunk.content for chunk in tables))
        self.assertTrue(all(chunk.metadata["extraction"] == "pdfplumber" for chunk in tables))

    def test_csv_is_indexed_as_table(self):
        chunks = self.ingestor.ingest("sales.csv", b"year,revenue\n2025,128")

        self.assertEqual(chunks[0].modality, "table")
        self.assertIn("year | revenue", chunks[0].content)

    def test_vision_caption_creates_chart_modality(self):
        vision_ingestor = DocumentIngestor(
            TokenChunker(500, 100, RegexTokenEncoding()),
            IngestionCache(self.temp_dir.name),
            FakeVisionAnalyzer(),
        )
        image = Image.new("RGB", (20, 20), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        chunks = vision_ingestor.ingest("chart.png", buffer.getvalue())

        self.assertEqual(chunks[0].modality, "chart")
        self.assertTrue(chunks[0].metadata["vision_analyzed"])

    def test_unsupported_file_type_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ingestor.ingest("archive.zip", b"not a document")


class GenerationTests(unittest.TestCase):
    def test_local_answer_contains_citation_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FaissVectorStore(FakeEmbeddingProvider(), directory)
            registry = DocumentRegistry(store)
            registry.initialize_demo()
            hits = HybridRetriever(store, fake_reranker()).search("revenue trend", limit=2)

            answer, model = AnswerGenerator().generate("What is the revenue trend?", hits)

        self.assertIn("[1]", answer)
        self.assertEqual(model, "local-grounded-synthesizer")


if __name__ == "__main__":
    unittest.main()
