from __future__ import annotations

import math
import re
from collections import Counter

from app.domain import DocumentChunk, SearchHit
from app.services.reranker import Reranker
from app.services.vector_store import FaissVectorStore

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class HybridRetriever:
    def __init__(
        self,
        vector_store: FaissVectorStore,
        reranker: Reranker,
        candidate_limit: int = 10,
    ) -> None:
        self.vector_store = vector_store
        self.reranker = reranker
        self.candidate_limit = candidate_limit

    def search(
        self,
        query: str,
        limit: int = 4,
        document_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        candidate_limit = max(limit, self.candidate_limit)
        semantic = self.vector_store.search(query, candidate_limit, document_ids)
        semantic_scores = {chunk.id: max(score, 0.0) for chunk, score in semantic}

        eligible = [
            chunk
            for chunk in self.vector_store.chunks
            if not document_ids or chunk.document_id in document_ids
        ]
        lexical_scores = self._bm25(query, eligible)
        chunks_by_id = {chunk.id: chunk for chunk in eligible}

        candidate_ids = set(semantic_scores) | set(
            sorted(lexical_scores, key=lexical_scores.get, reverse=True)[:candidate_limit]
        )
        hits = [
            SearchHit(
                chunk=chunks_by_id[chunk_id],
                semantic_score=semantic_scores.get(chunk_id, 0.0),
                lexical_score=lexical_scores.get(chunk_id, 0.0),
            )
            for chunk_id in candidate_ids
        ]

        candidates = sorted(hits, key=lambda hit: hit.score, reverse=True)[:candidate_limit]
        return self.reranker.rerank(query, candidates)[:limit]

    @staticmethod
    def _bm25(query: str, chunks: list[DocumentChunk]) -> dict[str, float]:
        if not chunks:
            return {}
        query_tokens = tokenize(query)
        tokenized = [tokenize(chunk.content) for chunk in chunks]
        avg_length = sum(map(len, tokenized)) / len(tokenized) or 1.0
        document_frequency = Counter(
            token for tokens in tokenized for token in set(tokens)
        )
        raw_scores: dict[str, float] = {}
        k1, b = 1.5, 0.75
        for chunk, tokens in zip(chunks, tokenized):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                idf = math.log(1 + (len(chunks) - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
                denominator = frequency + k1 * (1 - b + b * len(tokens) / avg_length)
                score += idf * frequency * (k1 + 1) / denominator
            raw_scores[chunk.id] = score

        maximum = max(raw_scores.values(), default=0.0) or 1.0
        return {chunk_id: score / maximum for chunk_id, score in raw_scores.items()}
