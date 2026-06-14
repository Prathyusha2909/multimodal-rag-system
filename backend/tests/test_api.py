import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.chunking import RegexTokenEncoding, TokenChunker
from tests.fakes import FakeEmbeddingProvider, fake_reranker


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        settings = Settings(
            frontend_origin="http://testserver",
            frontend_origin_regex=r"https://.*\.vercel\.app",
            upload_dir=root / "uploads",
            cache_dir=root / "cache",
            index_dir=root / "index",
        )
        cls.client = TestClient(
            create_app(
                settings,
                FakeEmbeddingProvider(),
                fake_reranker(),
                TokenChunker(500, 100, RegexTokenEncoding()),
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_health_and_stats(self):
        health = self.client.get("/health")
        stats = self.client.get("/api/v1/stats")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")
        self.assertEqual(stats.status_code, 200)
        self.assertEqual(stats.json()["index_backend"], "faiss:deterministic-test-embedding")

    def test_vercel_preview_origin_is_allowed(self):
        response = self.client.options(
            "/api/v1/query",
            headers={
                "Origin": "https://multimodal-rag-preview.vercel.app",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "https://multimodal-rag-preview.vercel.app",
        )

    def test_query_returns_ranked_citations(self):
        response = self.client.post(
            "/api/v1/query",
            json={"question": "What is the revenue trend in Figure 3?", "top_k": 3},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("$128M", payload["answer"])
        self.assertEqual(payload["citations"][0]["modality"], "chart")
        self.assertEqual(payload["citations"][0]["page"], 8)

    def test_text_upload_is_indexed(self):
        response = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("brief.txt", b"Customer retention rose to 91 percent.", "text/plain")},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["name"], "brief.txt")


if __name__ == "__main__":
    unittest.main()
