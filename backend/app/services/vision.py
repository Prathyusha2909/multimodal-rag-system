from __future__ import annotations

import json

from app.domain import Modality


class GeminiVisionAnalyzer:
    def __init__(self, api_key: str | None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model = model

    def analyze(self, content: bytes, mime_type: str) -> tuple[Modality, str] | None:
        if not self.api_key:
            return None
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        prompt = (
            "Analyze this document image for retrieval. Return JSON with keys modality and description. "
            "modality must be one of chart, table, or image. For charts include axes, series, values, "
            "and trend. For tables include headers and important row values. For other images provide "
            "a factual caption. Do not infer facts that are not visible."
        )
        response = client.models.generate_content(
            model=self.model,
            contents=[prompt, types.Part.from_bytes(data=content, mime_type=mime_type)],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        if not response.text:
            return None
        payload = json.loads(response.text)
        modality = payload.get("modality", "image")
        if modality not in {"chart", "table", "image"}:
            modality = "image"
        description = str(payload.get("description", "")).strip()
        return (modality, description) if description else None
