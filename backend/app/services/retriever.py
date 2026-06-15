from __future__ import annotations

import math
import re
from collections import Counter

from app.domain import DocumentChunk, SearchHit
from app.services.reranker import Reranker
from app.services.vector_store import FaissVectorStore

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SUMMARY_PATTERN = re.compile(r"\b(summarize|summary|overview|key findings|main points)\b", re.I)
SUMMARY_TERMS = {
    "achievement",
    "certification",
    "conclusion",
    "education",
    "experience",
    "finding",
    "objective",
    "overview",
    "profile",
    "project",
    "qualification",
    "requirement",
    "responsibility",
    "result",
    "role",
    "skill",
    "summary",
}
SUMMARY_EXPANSION = (
    "overview profile main topics key findings responsibilities requirements skills "
    "experience education projects achievements results conclusions"
)
BOILERPLATE_PHRASES = (
    "equal opportunity employer",
    "regardless of their",
    "subject to discrimination",
    "all rights reserved",
    "terms and conditions",
    "privacy policy",
)


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
        eligible = [
            chunk
            for chunk in self.vector_store.chunks
            if not document_ids or chunk.document_id in document_ids
        ]
        if SUMMARY_PATTERN.search(query):
            eligible = self._filter_requested_pages(query, eligible)
            return self._search_summary(query, eligible, limit, document_ids)

        candidate_limit = max(limit, self.candidate_limit)
        semantic = self.vector_store.search(query, candidate_limit, document_ids)
        semantic_scores = {chunk.id: max(score, 0.0) for chunk, score in semantic}
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

    def _search_summary(
        self,
        query: str,
        eligible: list[DocumentChunk],
        limit: int,
        document_ids: set[str] | None,
    ) -> list[SearchHit]:
        if not eligible:
            return []

        candidate_limit = max(limit, self.candidate_limit)
        expanded_query = f"{query} {SUMMARY_EXPANSION}"
        semantic = self.vector_store.search(
            expanded_query,
            candidate_limit * 2,
            document_ids,
        )
        eligible_ids = {chunk.id for chunk in eligible}
        semantic_scores = {
            chunk.id: max(score, 0.0)
            for chunk, score in semantic
            if chunk.id in eligible_ids
        }
        summary_scores = {
            chunk.id: self._summary_relevance(chunk)
            for chunk in eligible
        }
        maximum = max(summary_scores.values(), default=0.0) or 1.0
        summary_scores = {
            chunk_id: max(score, 0.0) / maximum
            for chunk_id, score in summary_scores.items()
        }
        chunks_by_id = {chunk.id: chunk for chunk in eligible}
        ranked_ids = sorted(
            chunks_by_id,
            key=lambda chunk_id: (
                0.6 * semantic_scores.get(chunk_id, 0.0)
                + 0.4 * summary_scores.get(chunk_id, 0.0)
            ),
            reverse=True,
        )[:candidate_limit]
        reranked = []
        for chunk_id in ranked_ids:
            semantic_score = semantic_scores.get(chunk_id, 0.0)
            summary_score = summary_scores.get(chunk_id, 0.0)
            hit = SearchHit(
                chunk=chunks_by_id[chunk_id],
                semantic_score=semantic_score,
                lexical_score=summary_score,
            )
            # Broad summaries use lightweight section-aware scoring. Loading the
            # cross-encoder here adds memory pressure without a focused query.
            hit.rerank_score = (
                0.6 * semantic_score
                + 0.4 * summary_score
                - self._boilerplate_penalty(hit.chunk.content)
            )
            reranked.append(hit)
        return self._select_page_diverse(
            sorted(reranked, key=lambda hit: hit.rerank_score, reverse=True),
            limit,
        )

    @staticmethod
    def _filter_requested_pages(query: str, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        match = re.search(r"\bpages?\s+(\d+)\s*(?:-|–|to)\s*(\d+)\b", query, re.I)
        if not match:
            return chunks
        start, end = sorted((int(match.group(1)), int(match.group(2))))
        filtered = [chunk for chunk in chunks if start <= chunk.page <= end]
        return filtered or chunks

    @staticmethod
    def _summary_relevance(chunk: DocumentChunk) -> float:
        text = f"{chunk.title} {chunk.content}".lower()
        tokens = set(tokenize(text))
        salience = len(tokens & SUMMARY_TERMS)
        heading_bonus = 1.5 if any(
            term in chunk.title.lower()
            for term in SUMMARY_TERMS
        ) else 0.0
        fact_bonus = min(len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text)), 4) * 0.15
        length_bonus = 0.5 if 20 <= len(tokenize(chunk.content)) <= 220 else 0.0
        return salience + heading_bonus + fact_bonus + length_bonus

    @classmethod
    def _boilerplate_penalty(cls, text: str) -> float:
        lower = text.lower()
        matches = sum(phrase in lower for phrase in BOILERPLATE_PHRASES)
        if matches >= 2:
            return 1.5
        if matches == 1 and len(tokenize(text)) < 180:
            return 0.9
        return 0.0

    @staticmethod
    def _select_page_diverse(hits: list[SearchHit], limit: int) -> list[SearchHit]:
        selected: list[SearchHit] = []
        selected_ids: set[str] = set()
        pages: set[tuple[str, int]] = set()
        for hit in hits:
            page_key = (hit.chunk.document_id, hit.chunk.page)
            if page_key in pages or hit.rerank_score <= 0:
                continue
            selected.append(hit)
            selected_ids.add(hit.chunk.id)
            pages.add(page_key)
            if len(selected) == limit:
                return selected
        for hit in hits:
            if hit.chunk.id in selected_ids or hit.rerank_score <= 0:
                continue
            selected.append(hit)
            if len(selected) == limit:
                break
        return selected

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
