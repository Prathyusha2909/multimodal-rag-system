from __future__ import annotations

import re
from pathlib import Path

from app.domain import SearchHit

TOKEN_PATTERN = re.compile(r"[a-z0-9%$]+")
VALUE_PATTERN = r"\$?\d+(?:\.\d+)?\s*(?:%|[KMB]|million|billion)?"
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "did",
    "do",
    "does",
    "explain",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "shown",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}


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
            "Answer the question directly and only from the supplied evidence. "
            "Synthesize the facts instead of repeating the evidence verbatim. "
            "Cite each claim with [1], [2], etc. If evidence is incomplete, say so.\n\n"
            f"Question: {question}\n\nEvidence:\n{context}"
        )
        response = client.models.generate_content(model=self.model, contents=prompt)
        return response.text or "No grounded answer was generated."

    @classmethod
    def _generate_locally(cls, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return "I could not find relevant evidence in the indexed documents."

        trend_answer = cls._trend_answer(question, hits[0])
        if trend_answer:
            return trend_answer

        comparison = cls._metric_comparison(question, hits[0])
        if comparison:
            return comparison

        return cls._query_focused_summary(question, hits)

    @classmethod
    def _trend_answer(cls, question: str, hit: SearchHit) -> str | None:
        if not re.search(r"\b(trend|change|changed|increase|increased|decrease|decreased|growth|grew)\b", question, re.I):
            return None

        text = cls._clean_content(hit)
        pairs = cls._year_value_pairs(text)
        if len(pairs) < 2:
            return None

        first_year, first_value = pairs[0]
        last_year, last_value = pairs[-1]
        direction = "changed"
        first_number = cls._numeric_value(first_value)
        last_number = cls._numeric_value(last_value)
        if last_number > first_number:
            direction = "increased"
        elif last_number < first_number:
            direction = "decreased"

        subject = cls._metric_subject(question)
        answer = (
            f"{subject} {direction} from {first_value} in {first_year} to "
            f"{last_value} in {last_year} [1]."
        )
        detail = cls._supporting_detail(text)
        if detail and detail.lower() not in answer.lower():
            answer += f" {detail} [1]."
        return answer

    @classmethod
    def _metric_comparison(cls, question: str, hit: SearchHit) -> str | None:
        if "compare" not in question.lower():
            return None

        metric_match = re.search(r"\b(revenue|profit|margin|sales|emissions?)\b", question, re.I)
        if not metric_match:
            return None
        metric = metric_match.group(1).lower()
        query_prefix = question[: metric_match.start()]
        query_prefix = re.sub(r"^.*?\bcompare\b", "", query_prefix, flags=re.I)
        entities = [
            re.sub(r"\b(?:the|segment|figure|table)\b|\d+", "", item, flags=re.I).strip()
            for item in re.split(r"\band\b|,|\bversus\b|\bvs\.?\b", query_prefix, flags=re.I)
        ]
        entities = [entity for entity in entities if entity]
        if len(entities) != 2:
            return None

        text = cls._clean_content(hit)
        values = [cls._entity_metric_value(text, entity, metric, entities) for entity in entities]
        if any(value is None for value in values):
            return None

        first_value, second_value = values
        return (
            f"{entities[0].title()} {metric} is {first_value}, compared with "
            f"{second_value} for {entities[1].title()} [1]."
        )

    @classmethod
    def _query_focused_summary(cls, question: str, hits: list[SearchHit]) -> str:
        query_tokens = cls._meaningful_tokens(question)
        candidates: list[tuple[float, int, int, str]] = []
        for citation, hit in enumerate(hits, start=1):
            text = cls._clean_content(hit)
            for sentence_index, sentence in enumerate(cls._sentences(text)):
                sentence_tokens = cls._meaningful_tokens(sentence)
                overlap = len(query_tokens & sentence_tokens)
                if not overlap and citation > 1:
                    continue
                numeric_bonus = cls._answer_type_bonus(question, sentence)
                generic_visual_penalty = 2.0 if not re.search(r"\d", sentence) and re.search(
                    r"\b(?:graph|chart)\s+(?:compares|shows)|\bline\s+(?:graph|chart)\b",
                    sentence,
                    re.I,
                ) else 0.0
                score = overlap * 2.0 + numeric_bonus - generic_visual_penalty + 1.0 / citation - sentence_index * 0.05
                candidates.append((score, citation, sentence_index, sentence))

        selected: list[tuple[int, int, str]] = []
        selection_limit = 3 if re.search(r"\b(summarize|summary|findings)\b", question, re.I) else 1
        if re.search(r"\b(explain|relationship|related)\b", question, re.I):
            selection_limit = 2
        seen: set[str] = set()
        for _, citation, sentence_index, sentence in sorted(candidates, reverse=True):
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            if key in seen or any(cls._near_duplicate(key, prior) for prior in seen):
                continue
            selected.append((citation, sentence_index, sentence))
            seen.add(key)
            if len(selected) == selection_limit:
                break

        if not selected:
            return "I could not find sufficiently relevant evidence in the selected documents."
        selected.sort(key=lambda item: (item[0], item[1]))
        return " ".join(
            f"{cls._ensure_sentence(sentence)} [{citation}]"
            for citation, _, sentence in selected
        )

    @staticmethod
    def _clean_content(hit: SearchHit) -> str:
        text = hit.chunk.content.replace("\n", " ")
        document_name = Path(hit.chunk.document_name).stem.replace("_", " ")
        text = re.sub(r"\bSYNTHETIC SAMPLE REPORT\b", "", text, flags=re.I)
        text = re.sub(re.escape(document_name), "", text, flags=re.I)
        text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.I)
        text = re.sub(r"\$\s+(?=\d)", "$", text)
        text = re.sub(r"(\d)\.\s+(\d)(?=\s*%)", r"\1.\2", text)
        text = re.sub(r"\s+", " ", text).strip(" -")
        return text

    @classmethod
    def _year_value_pairs(cls, text: str) -> list[tuple[str, str]]:
        patterns = [
            rf"(?P<year>(?:19|20)\d{{2}})\s*[:\-]?\s*(?P<value>{VALUE_PATTERN})",
            rf"(?P<value>{VALUE_PATTERN})\s*(?:in|during|by)\s*(?P<year>(?:19|20)\d{{2}})",
        ]
        by_year: dict[str, tuple[int, str]] = {}
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.I):
                value = re.sub(r"\s+", "", match.group("value"))
                if not re.search(r"(?:\$|%|[KMB]$|million$|billion$)", value, re.I):
                    continue
                year = match.group("year")
                current = by_year.get(year)
                if current is None or match.start() < current[0]:
                    by_year[year] = (match.start(), value)
        return [(year, by_year[year][1]) for year in sorted(by_year)]

    @staticmethod
    def _numeric_value(value: str) -> float:
        match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
        return float(match.group()) if match else 0.0

    @staticmethod
    def _metric_subject(question: str) -> str:
        subjects = (
            "renewable electricity adoption",
            "operating profit",
            "annual revenue",
            "revenue",
            "sales",
            "profit",
            "customer growth",
        )
        lower = question.lower()
        for subject in subjects:
            if subject in lower:
                return subject.capitalize()
        return "The reported metric"

    @classmethod
    def _supporting_detail(cls, text: str) -> str | None:
        largest = re.search(r"(?:with\s+)?the\s+largest[^.!?]+", text, re.I)
        if largest:
            detail = re.sub(r"^with\s+", "", largest.group(), flags=re.I)
            return detail[0].upper() + detail[1:]
        for sentence in cls._sentences(text):
            if re.search(r"\b(faster|target|margin)\b", sentence, re.I):
                sentence = re.sub(r"^(?:Figure|Chart)\s+\d+\s*(?:is|-)?\s*", "", sentence, flags=re.I)
                return sentence.rstrip(".")
        return None

    @classmethod
    def _entity_metric_value(
        cls,
        text: str,
        entity: str,
        metric: str,
        all_entities: list[str],
    ) -> str | None:
        entity_pattern = rf"\b{re.escape(entity)}s?\b"
        entity_match = re.search(entity_pattern, text, re.I)
        if not entity_match:
            return None

        endpoint = len(text)
        for other in all_entities:
            if other == entity:
                continue
            match = re.search(rf"\b{re.escape(other)}s?\b", text[entity_match.end():], re.I)
            if match:
                endpoint = min(endpoint, entity_match.end() + match.start())
        segment = text[entity_match.end():endpoint]

        direct = re.search(rf"\b{re.escape(metric)}\b\s*[:\-]?\s*({VALUE_PATTERN})", segment, re.I)
        if direct:
            return re.sub(r"\s+", "", direct.group(1))

        headers = [name for name in ("revenue", "profit", "margin", "sales", "emissions") if re.search(rf"\b{name}\b", text[:entity_match.start()], re.I)]
        if metric not in headers:
            return None
        values = [re.sub(r"\s+", "", value) for value in re.findall(VALUE_PATTERN, segment, re.I) if re.search(r"\d", value)]
        metric_index = headers.index(metric)
        return values[metric_index] if metric_index < len(values) else None

    @classmethod
    def _sentences(cls, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [part.strip(" -") for part in parts if len(part.strip()) > 2]

    @classmethod
    def _meaningful_tokens(cls, text: str) -> set[str]:
        tokens = set()
        for token in TOKEN_PATTERN.findall(text.lower()):
            if token in STOP_WORDS or len(token) < 2:
                continue
            tokens.add(token[:-1] if token.endswith("s") and len(token) > 4 else token)
        return tokens

    @staticmethod
    def _answer_type_bonus(question: str, sentence: str) -> float:
        if re.search(r"\b(share|percent|percentage)\b", question, re.I):
            return 5.0 if "%" in sentence else 0.0
        if re.search(r"\bhow many\b", question, re.I):
            without_years = re.sub(r"\b(?:19|20)\d{2}\b", "", sentence)
            return 5.0 if re.search(r"\d", without_years) else 0.0
        if re.search(r"\b(compare|relationship|related|amount|total)\b", question, re.I):
            return 3.0 if re.search(r"\d", sentence) else 0.0
        return 0.0

    @staticmethod
    def _near_duplicate(candidate: str, existing: str) -> bool:
        candidate_tokens = set(candidate.split())
        existing_tokens = set(existing.split())
        union = candidate_tokens | existing_tokens
        return bool(union) and len(candidate_tokens & existing_tokens) / len(union) > 0.8

    @staticmethod
    def _ensure_sentence(text: str) -> str:
        text = text.strip()
        return text if text.endswith((".", "!", "?")) else f"{text}."
