import hashlib
import re

import numpy as np

from app.services.reranker import HeuristicReranker


class FakeEmbeddingProvider:
    name = "deterministic-test-embedding"
    dimension = 128

    def embed_documents(self, texts):
        return np.vstack([self._embed(text) for text in texts]).astype("float32")

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        vector = np.zeros(self.dimension, dtype="float32")
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            vector[value % self.dimension] += 1.0
        norm = np.linalg.norm(vector) or 1.0
        return vector / norm


class FakeVisionAnalyzer:
    api_key = "test"
    model = "fake-vision"

    def analyze(self, content, mime_type):
        return "chart", "A bar chart shows revenue increasing from 10 to 20."


class NoVisionAnalyzer:
    api_key = None
    model = "disabled"

    def analyze(self, content, mime_type):
        return None


def fake_reranker():
    return HeuristicReranker()
