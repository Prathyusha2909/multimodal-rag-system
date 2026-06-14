from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from app.domain import DocumentChunk
from app.services.cache import IngestionCache
from app.services.chunking import TokenChunker
from app.services.vision import GeminiVisionAnalyzer

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


class DocumentIngestor:
    def __init__(
        self,
        chunker: TokenChunker,
        cache: IngestionCache,
        vision: GeminiVisionAnalyzer,
    ) -> None:
        self.chunker = chunker
        self.cache = cache
        self.vision = vision

    def ingest(self, filename: str, content: bytes) -> list[DocumentChunk]:
        suffix = Path(filename).suffix.lower()
        document_id = hashlib.sha256(content).hexdigest()[:16]
        cache_key = self._cache_key(content, suffix, filename)
        cached = self.cache.load(cache_key)
        if cached is not None:
            return cached

        if suffix == ".pdf":
            chunks = self._ingest_pdf(document_id, filename, content)
        elif suffix in IMAGE_MIME_TYPES:
            chunks = self._ingest_image(document_id, filename, content, IMAGE_MIME_TYPES[suffix])
        elif suffix in {".txt", ".md"}:
            text = content.decode("utf-8", errors="replace")
            chunks = self.chunker.split(document_id, filename, 1, text)
        elif suffix == ".csv":
            chunks = self._ingest_csv(document_id, filename, content)
        else:
            raise ValueError("Supported formats: PDF, PNG, JPG, WEBP, TIFF, TXT, MD, and CSV")

        self.cache.save(cache_key, chunks)
        return chunks

    def _ingest_pdf(self, document_id: str, filename: str, content: bytes) -> list[DocumentChunk]:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        chunks: list[DocumentChunk] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                chunks.extend(self.chunker.split(document_id, filename, page_number, text))
            else:
                chunks.append(
                    DocumentChunk(
                        id=f"{document_id}-{page_number}-scan-1",
                        document_id=document_id,
                        document_name=filename,
                        page=page_number,
                        modality="scan",
                        title=f"Scanned page {page_number}",
                        content="This PDF page has no extractable text and requires OCR or vision analysis.",
                        metadata={"filename": filename, "page": page_number, "chunk_index": 1},
                    )
                )

        chunks.extend(self._extract_pdf_tables(document_id, filename, content))
        return chunks

    def _extract_pdf_tables(self, document_id: str, filename: str, content: bytes) -> list[DocumentChunk]:
        import pdfplumber

        chunks: list[DocumentChunk] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                for table_index, table in enumerate(page.extract_tables(), start=1):
                    table_text = self._table_to_markdown(table)
                    if not table_text:
                        continue
                    chunks.extend(
                        self.chunker.split(
                            document_id,
                            filename,
                            page_number,
                            table_text,
                            modality="table",
                            title=f"Table {table_index} on page {page_number}",
                            id_prefix=f"{document_id}-{page_number}-table-{table_index}",
                            metadata={"table_index": table_index, "extraction": "pdfplumber"},
                        )
                    )
        return chunks

    def _ingest_image(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        mime_type: str,
    ) -> list[DocumentChunk]:
        ocr_text = self._ocr_image(content)
        vision_result = None
        try:
            vision_result = self.vision.analyze(content, mime_type)
        except Exception:
            vision_result = None

        modality = vision_result[0] if vision_result else "image"
        parts = []
        if vision_result:
            parts.append(f"Visual description: {vision_result[1]}")
        if ocr_text:
            parts.append(f"OCR text: {ocr_text}")
        if not parts:
            parts.append("Image indexed without a generated caption or OCR text.")

        return self.chunker.split(
            document_id,
            filename,
            1,
            "\n\n".join(parts),
            modality=modality,
            title="Uploaded visual",
            id_prefix=f"{document_id}-1-{modality}",
            metadata={
                "mime_type": mime_type,
                "ocr_available": bool(ocr_text),
                "vision_analyzed": bool(vision_result),
            },
        )

    def _ingest_csv(self, document_id: str, filename: str, content: bytes) -> list[DocumentChunk]:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8", errors="replace"))))
        table_text = self._table_to_markdown(rows)
        return self.chunker.split(
            document_id,
            filename,
            1,
            table_text,
            modality="table",
            title="CSV table",
            id_prefix=f"{document_id}-1-table",
            metadata={"extraction": "csv"},
        )

    @staticmethod
    def _ocr_image(content: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image

            return pytesseract.image_to_string(Image.open(io.BytesIO(content))).strip()
        except (ImportError, OSError, RuntimeError):
            return ""

    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        rows = [[str(cell or "").strip() for cell in row] for row in table if row]
        rows = [row for row in rows if any(row)]
        if not rows:
            return ""
        width = max(map(len, rows))
        rows = [row + [""] * (width - len(row)) for row in rows]
        header = rows[0]
        lines = [" | ".join(header), " | ".join(["---"] * width)]
        lines.extend(" | ".join(row) for row in rows[1:])
        return "\n".join(lines)

    def _cache_key(self, content: bytes, suffix: str, filename: str) -> str:
        fingerprint = json.dumps(
            {
                "sha256": hashlib.sha256(content).hexdigest(),
                "suffix": suffix,
                "filename": filename,
                "chunk_size": self.chunker.chunk_size,
                "overlap": self.chunker.overlap,
                "vision": bool(self.vision.api_key),
                "vision_model": self.vision.model,
                "version": 2,
            },
            sort_keys=True,
        )
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
