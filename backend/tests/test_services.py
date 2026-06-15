import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.services.cache import IngestionCache
from app.services.chunking import RegexTokenEncoding, TokenChunker
from app.services.generator import AnswerGenerator
from app.domain import DocumentChunk, SearchHit
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

    def test_local_answer_synthesizes_raw_pdf_trend(self):
        hit = SearchHit(
            chunk=DocumentChunk(
                id="raw-chart",
                document_id="nova-upload",
                document_name="Nova_Retail_Annual_Report_2025.pdf",
                page=8,
                modality="text",
                content=(
                    "SYNTHETIC SAMPLE REPORT Nova Retail Annual Report 2025 Page 8 "
                    "Figure 3 - Annual Revenue Trend 2021 $ 78M 2022 $ 86M 2023 $ 97M "
                    "2024 $ 108M 2025 $ 128M Revenue increased in each reported year, "
                    "with the largest annual increase of $ 20M occurring in 2025."
                ),
            ),
            semantic_score=0.9,
            lexical_score=1.0,
        )

        answer, _ = AnswerGenerator().generate("What is the revenue trend shown in Figure 3?", [hit])

        self.assertIn("increased from $78M in 2021 to $128M in 2025", answer)
        self.assertIn("largest annual increase was $20M", answer)
        self.assertNotIn("SYNTHETIC SAMPLE REPORT", answer)

    def test_local_answer_compares_table_metric(self):
        hit = SearchHit(
            chunk=DocumentChunk(
                id="raw-table",
                document_id="nova-upload",
                document_name="Nova_Retail_Annual_Report_2025.pdf",
                page=9,
                modality="table",
                content=(
                    "Table 2 - Segment Performance Segment Revenue Profit Margin "
                    "Online $ 62M $ 15M 24. 2% Stores $ 51M $ 8M 15. 7%"
                ),
            ),
            semantic_score=0.9,
            lexical_score=1.0,
        )

        answer, _ = AnswerGenerator().generate("Compare online and store profit in Table 2.", [hit])

        self.assertIn("Online profit is $15M", answer)
        self.assertIn("$8M for Stores", answer)

    def test_local_summary_excludes_equal_opportunity_boilerplate(self):
        chunks = [
            DocumentChunk(
                id="disclaimer",
                document_id="job",
                document_name="Data_Scientist.pdf",
                page=1,
                modality="text",
                content=(
                    "We are an equal opportunity employer committed to promoting all talents "
                    "regardless of age or any characteristic subject to discrimination."
                ),
            ),
            DocumentChunk(
                id="overview",
                document_id="job",
                document_name="Data_Scientist.pdf",
                page=2,
                modality="text",
                content=(
                    "The Associate Data Scientist analyzes business data and develops "
                    "machine-learning solutions for stakeholders."
                ),
            ),
            DocumentChunk(
                id="skills",
                document_id="job",
                document_name="Data_Scientist.pdf",
                page=3,
                modality="text",
                content=(
                    "Required skills include Python, SQL, statistics, and data visualization."
                ),
            ),
        ]
        hits = [SearchHit(chunk, 0.8, 0.8) for chunk in chunks]

        answer, _ = AnswerGenerator().generate("summarize", hits)

        self.assertIn("Associate Data Scientist", answer)
        self.assertIn("Python", answer)
        self.assertNotIn("equal opportunity", answer.lower())

    def test_job_description_summary_extracts_role_skills_and_eligibility(self):
        chunks = [
            DocumentChunk(
                id="role",
                document_id="job",
                document_name="DAI_Associate data scientist_2026_YG.pdf",
                page=1,
                modality="text",
                content=(
                    "We are an equal opportunity employer. Role: Associate data scientist "
                    "Candidates should have responsibilities: • Assist in developing and optimizing "
                    "Generative AI models • Collaborate with teams to integrate AI solutions "
                    "• Research emerging AI trends • Establishing data preparation best practices"
                ),
            ),
            DocumentChunk(
                id="skills",
                document_id="job",
                document_name="DAI_Associate data scientist_2026_YG.pdf",
                page=2,
                modality="text",
                content=(
                    "Required Skillset • Generative AI and Machine Learning • NLP, Python, Java, "
                    "React, Node.js, MySQL, MongoDB • Communication and teamwork. Eligibility "
                    "Criteria: • 2026 passed outs with no backlogs • Bachelor's degree in Computer "
                    "Science, AI, or Data Science • Minimum of 70% across 10th, 12th and Graduation"
                ),
            ),
        ]
        hits = [SearchHit(chunk, 0.8, 0.8) for chunk in chunks]

        answer, _ = AnswerGenerator().generate("summarize", hits)

        self.assertIn("Associate Data Scientist opportunity", answer)
        self.assertIn("Generative AI", answer)
        self.assertIn("Python", answer)
        self.assertIn("2026 graduates", answer)
        self.assertIn("at least 70%", answer)
        self.assertNotIn("equal opportunity", answer.lower())


if __name__ == "__main__":
    unittest.main()
