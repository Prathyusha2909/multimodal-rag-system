from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Protocol

from app.domain import SearchHit

os.environ.setdefault("USE_TF", "0")


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]: ...

    def release_model(self) -> None: ...


class HeuristicReranker:
    name = "heuristic-fallback"

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        query_tokens = set(_tokenize(query))
        modality_terms = {
            "chart": {"chart", "figure", "graph", "trend", "plot"},
            "table": {"table", "row", "column", "compare"},
            "image": {"image", "photo", "diagram"},
            "scan": {"scan", "scanned", "handwritten"},
        }
        for hit in hits:
            overlap = len(query_tokens & set(_tokenize(hit.chunk.content))) / max(len(query_tokens), 1)
            modality_boost = 0.1 if query_tokens & modality_terms.get(hit.chunk.modality, set()) else 0.0
            hit.rerank_score = 0.5 * hit.semantic_score + 0.3 * hit.lexical_score + 0.2 * overlap + modality_boost
        return sorted(hits, key=lambda hit: hit.rerank_score, reverse=True)

    def release_model(self) -> None:
        return None


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path | str) -> None:
        self.name = model_name
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.model_cache_dir = self.cache_dir / "models" / "sentence-transformers"
        model_key = hashlib.sha256(
            f"sentence-transformers:{model_name}".encode("utf-8")
        ).hexdigest()[:12]
        self.score_cache_path = self.cache_dir / "reranker" / f"{model_key}.json"
        self.score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._fallback = HeuristicReranker()
        self._scores = self._load_scores()

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        try:
            raw_scores = self._score(query, [hit.chunk.content for hit in hits])
        except Exception:
            return self._fallback.rerank(query, hits)

        minimum, maximum = min(raw_scores), max(raw_scores)
        span = maximum - minimum
        normalized = [0.5 if span == 0 else (score - minimum) / span for score in raw_scores]
        for hit, raw, cross_score in zip(hits, raw_scores, normalized):
            hit.cross_encoder_score = float(raw)
            hit.rerank_score = 0.75 * cross_score + 0.15 * hit.semantic_score + 0.1 * hit.lexical_score
        return sorted(hits, key=lambda hit: hit.rerank_score, reverse=True)

    def _score(self, query: str, documents: list[str]) -> list[float]:
        values: list[float | None] = [None] * len(documents)
        missing_documents = []
        missing_indices = []
        for index, document in enumerate(documents):
            key = self._key(query, document)
            if key in self._scores:
                values[index] = self._scores[key]
            else:
                missing_documents.append(document)
                missing_indices.append(index)
        if missing_documents:
            pairs = [(query, document) for document in missing_documents]
            generated = self._get_model().predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
            )
            for index, document, score in zip(missing_indices, missing_documents, generated):
                value = float(score)
                values[index] = value
                self._scores[self._key(query, document)] = value
            self.score_cache_path.write_text(json.dumps(self._scores), encoding="utf-8")
        return [float(value) for value in values]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                cache_dir=str(self.model_cache_dir),
                local_files_only=self._has_cached_model(),
            )
        return self._model

    def release_model(self) -> None:
        if self._model is not None:
            self._model = None
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (AttributeError, OSError):
                pass

    def _has_cached_model(self) -> bool:
        model_dir = self.model_cache_dir / f"models--{self.model_name.replace('/', '--')}"
        snapshots = model_dir / "snapshots"
        return snapshots.exists() and any(snapshots.iterdir())

    def _load_scores(self) -> dict[str, float]:
        if not self.score_cache_path.exists():
            return {}
        try:
            return json.loads(self.score_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _key(query: str, document: str) -> str:
        return hashlib.sha256(f"{query}\0{document}".encode("utf-8")).hexdigest()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$%.]+", text.lower())
