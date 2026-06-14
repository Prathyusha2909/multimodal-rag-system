import tempfile
import unittest

import numpy as np

from app.services.embedding import SentenceTransformerProvider
from app.services.reranker import CrossEncoderReranker


class RecordingSentenceTransformer:
    def __init__(self) -> None:
        self.batch_sizes = []

    def encode(self, texts, *, batch_size, **kwargs):
        self.batch_sizes.append(batch_size)
        return np.ones((len(texts), 384), dtype="float32")


class ModelLifecycleTests(unittest.TestCase):
    def test_embedding_provider_uses_configured_small_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = SentenceTransformerProvider(cache_dir=directory, batch_size=3)
            model = RecordingSentenceTransformer()
            provider._model = model

            vectors = provider.embed_documents(["one", "two", "three", "four"])

            self.assertEqual(vectors.shape, (4, 384))
            self.assertEqual(model.batch_sizes, [3])

    def test_cross_encoder_model_can_be_released_before_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            reranker = CrossEncoderReranker("test-model", directory)
            reranker._model = object()

            reranker.release_model()

            self.assertIsNone(reranker._model)


if __name__ == "__main__":
    unittest.main()
