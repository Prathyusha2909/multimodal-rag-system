import unittest

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = Settings(frontend_origin="http://testserver")
        cls.client = TestClient(create_app(settings))

    def test_health_and_stats(self):
        health = self.client.get("/health")
        stats = self.client.get("/api/v1/stats")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")
        self.assertEqual(stats.status_code, 200)
        self.assertGreaterEqual(stats.json()["documents"], 2)

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
