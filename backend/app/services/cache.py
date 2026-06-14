from __future__ import annotations

import json
from pathlib import Path

from app.domain import DocumentChunk


class IngestionCache:
    def __init__(self, cache_dir: Path | str) -> None:
        self.root = Path(cache_dir) / "ingestion"
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> list[DocumentChunk] | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            return [DocumentChunk(**value) for value in values]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save(self, key: str, chunks: list[DocumentChunk]) -> None:
        (self.root / f"{key}.json").write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], indent=2),
            encoding="utf-8",
        )
