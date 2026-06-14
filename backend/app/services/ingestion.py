from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from app.domain import DocumentChunk


class DocumentIngestor:
    def ingest(self, filename: str, content: bytes) -> list[DocumentChunk]:
        suffix = Path(filename).suffix.lower()
        document_id = hashlib.sha1(content).hexdigest()[:12]
        if suffix == ".pdf":
            return self._ingest_pdf(document_id, filename, content)
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            return self._ingest_image(document_id, filename, content)
        if suffix in {".txt", ".md", ".csv"}:
            text = content.decode("utf-8", errors="replace")
            return self._chunk_text(document_id, filename, 1, text)
        raise ValueError("Supported formats: PDF, PNG, JPG, WEBP, TIFF, TXT, MD, and CSV")

    def _ingest_pdf(self, document_id: str, filename: str, content: bytes) -> list[DocumentChunk]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install backend dependencies to ingest PDFs") from exc

        reader = PdfReader(io.BytesIO(content))
        chunks: list[DocumentChunk] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                chunks.extend(self._chunk_text(document_id, filename, page_number, text))
            else:
                chunks.append(
                    DocumentChunk(
                        id=f"{document_id}-{page_number}-scan",
                        document_id=document_id,
                        document_name=filename,
                        page=page_number,
                        modality="scan",
                        title=f"Scanned page {page_number}",
                        content=(
                            "This page appears to be scanned and has no extractable text. "
                            "Convert the page to an image and upload it to use the OCR path."
                        ),
                    )
                )
        return chunks

    def _ingest_image(self, document_id: str, filename: str, content: bytes) -> list[DocumentChunk]:
        extracted = ""
        try:
            import pytesseract
            from PIL import Image

            extracted = pytesseract.image_to_string(Image.open(io.BytesIO(content))).strip()
        except (ImportError, OSError, RuntimeError):
            extracted = ""

        description = extracted or (
            "Image uploaded successfully, but OCR text was unavailable. Install the Tesseract "
            "binary locally to extract text from image uploads."
        )
        return [
            DocumentChunk(
                id=f"{document_id}-1-image",
                document_id=document_id,
                document_name=filename,
                page=1,
                modality="image",
                title="Uploaded image",
                content=description,
            )
        ]

    @staticmethod
    def _chunk_text(
        document_id: str,
        filename: str,
        page: int,
        text: str,
        max_words: int = 180,
    ) -> list[DocumentChunk]:
        words = re.split(r"\s+", text.strip())
        chunks = []
        for index in range(0, len(words), max_words):
            part = " ".join(words[index : index + max_words])
            if not part:
                continue
            chunk_number = index // max_words + 1
            chunks.append(
                DocumentChunk(
                    id=f"{document_id}-{page}-{chunk_number}",
                    document_id=document_id,
                    document_name=filename,
                    page=page,
                    modality="text",
                    title=f"Page {page}",
                    content=part,
                )
            )
        return chunks
