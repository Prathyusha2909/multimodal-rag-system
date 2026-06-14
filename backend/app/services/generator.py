from __future__ import annotations

import re

from app.domain import SearchHit


class AnswerGenerator:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, question: str, hits: list[SearchHit]) -> tuple[str, str]:
        if self.api_key:
            try:
                return self._generate_with_gemini(question, hits), self.model
            except Exception:
                # The demo remains available when the external model is rate-limited.
                pass
        return self._generate_locally(question, hits), "local-grounded-synthesizer"

    def _generate_with_gemini(self, question: str, hits: list[SearchHit]) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        context = "\n\n".join(
            f"[{index}] {hit.chunk.document_name}, page {hit.chunk.page}, "
            f"{hit.chunk.modality}: {hit.chunk.content}"
            for index, hit in enumerate(hits, start=1)
        )
        prompt = (
            "Answer only from the supplied evidence. Cite claims with [1], [2], etc. "
            "If evidence is incomplete, say so.\n\n"
            f"Question: {question}\n\nEvidence:\n{context}"
        )
        response = client.models.generate_content(model=self.model, contents=prompt)
        return response.text or "No grounded answer was generated."

    @staticmethod
    def _generate_locally(question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return "I could not find relevant evidence in the indexed documents."

        lower_question = question.lower()
        if "compare" in lower_question and len(hits) >= 2:
            first, second = hits[:2]
            return (
                f"The strongest comparison is between {first.chunk.title} and "
                f"{second.chunk.title}. {first.chunk.content} [{1}] "
                f"In comparison, {second.chunk.content} [{2}]"
            )

        sentences: list[str] = []
        for index, hit in enumerate(hits[:3], start=1):
            candidates = re.split(r"(?<=[.!?])\s+", hit.chunk.content)
            selected = " ".join(candidates[:2]).strip()
            if selected:
                sentences.append(f"{selected} [{index}]")
        return " ".join(sentences)
