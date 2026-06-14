from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from textwrap import wrap

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "demo" / "demo.mp4"
WIDTH = 1280
HEIGHT = 720
FPS = 12


@dataclass(frozen=True)
class Segment:
    duration: int
    title: str
    subtitle: str
    image: Path | None = None
    points: tuple[str, ...] = ()


SEGMENTS = (
    Segment(
        8,
        "Multimodal RAG System",
        "A reproducible portfolio project for cited document question answering",
        points=("Python + FastAPI", "React", "SentenceTransformers + FAISS"),
    ),
    Segment(
        14,
        "1. Upload mixed-format evidence",
        "The workspace accepts PDFs, scans, images, text, Markdown, and CSV files.",
        ROOT / "screenshots" / "home-page.png",
    ),
    Segment(
        15,
        "2. Extract each modality",
        "PDF text, tables, OCR output, and optional vision descriptions remain identifiable.",
        ROOT / "docs" / "workflow.png",
    ),
    Segment(
        15,
        "3. Build citation-ready chunks",
        "Chunks use 500 tokens, 100-token overlap, and filename, page, modality, and chunk metadata.",
        ROOT / "docs" / "architecture.png",
    ),
    Segment(
        15,
        "4. Retrieve and rerank evidence",
        "BGE embeddings and FAISS combine with BM25 before a CrossEncoder selects the best passages.",
        ROOT / "docs" / "system-design.png",
    ),
    Segment(
        15,
        "5. Ask a multimodal question",
        "The same query can compare narrative text, a chart description, and a table row.",
        ROOT / "screenshots" / "query-example.png",
    ),
    Segment(
        16,
        "6. Inspect the grounded result",
        "Every answer exposes the source document, page, modality, excerpt, and reranking score.",
        ROOT / "screenshots" / "result-example.png",
    ),
    Segment(
        12,
        "7. Evaluate the real retrieval path",
        "DeepEval custom metrics on an eight-question synthetic test set",
        points=("Retrieval Hit@4: 100%", "Cited-page coverage: 100%", "Required-fact coverage: 95.8%"),
    ),
    Segment(
        10,
        "8. Deploy as a portfolio demo",
        "Vercel serves the React interface while Render runs the containerized FastAPI backend.",
        points=("No private or company data", "Free-tier limitations documented", "Local mode remains fully supported"),
    ),
    Segment(
        6,
        "Explore the implementation",
        "Architecture, sample documents, evaluation outputs, tests, and setup instructions are all included.",
        points=("github.com/Prathyusha2909/multimodal-rag-system",),
    ),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


@lru_cache(maxsize=1)
def background_template() -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#07111d")
    pixels = canvas.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            glow = max(0.0, 1.0 - ((x - 1030) ** 2 + (y - 80) ** 2) ** 0.5 / 720)
            pixels[x, y] = (7 + int(10 * glow), 17 + int(19 * glow), 29 + int(24 * glow))
    return canvas


def background() -> Image.Image:
    return background_template().copy()


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], width: int, text_font, fill: str, spacing: int = 8) -> None:
    average_character_width = max(8, text_font.getlength("ABCDEFGHIJKLMNOPQRSTUVWXYZ") / 26)
    lines = wrap(text, width=max(10, int(width / average_character_width)))
    draw.multiline_text(xy, "\n".join(lines), font=text_font, fill=fill, spacing=spacing)


def image_frame(segment: Segment, progress: float) -> np.ndarray:
    canvas = background()
    draw = ImageDraw.Draw(canvas)
    draw.text((54, 35), segment.title, font=font(34, bold=True), fill="#b8ee45")
    draw_wrapped(draw, segment.subtitle, (56, 83), 1168, font(22), "#dbe8f5", 5)

    source = Image.open(segment.image).convert("RGB")
    max_width, max_height = 1168, 550
    scale = min(max_width / source.width, max_height / source.height) * (1.0 + 0.012 * progress)
    resized = source.resize((int(source.width * scale), int(source.height * scale)), Image.Resampling.LANCZOS)
    left = (WIDTH - resized.width) // 2
    top = 148 + (max_height - resized.height) // 2
    canvas.paste(resized, (left, top))
    draw.rounded_rectangle((left - 2, top - 2, left + resized.width + 2, top + resized.height + 2), radius=8, outline="#536f86", width=2)
    return np.asarray(canvas)


def card_frame(segment: Segment, progress: float) -> np.ndarray:
    canvas = background()
    draw = ImageDraw.Draw(canvas)
    accent_width = int(170 + 790 * min(progress * 3, 1))
    draw.rounded_rectangle((95, 96, 95 + accent_width, 103), radius=4, fill="#b8ee45")
    draw_wrapped(draw, segment.title, (95, 145), 1050, font(48, bold=True), "#f4f8fc", 10)
    draw_wrapped(draw, segment.subtitle, (98, 280), 1035, font(26), "#b9cadb", 8)

    y = 410
    for point in segment.points:
        draw.rounded_rectangle((98, y + 8, 111, y + 21), radius=3, fill="#b8ee45")
        draw_wrapped(draw, point, (132, y), 1010, font(24, bold=True), "#e4edf6", 5)
        y += 62
    return np.asarray(canvas)


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    with imageio.get_writer(OUTPUT, fps=FPS, codec="libx264", quality=7, macro_block_size=None) as writer:
        for segment in SEGMENTS:
            frame_count = segment.duration * FPS
            for index in range(frame_count):
                progress = index / max(frame_count - 1, 1)
                frame = image_frame(segment, progress) if segment.image else card_frame(segment, progress)
                writer.append_data(frame)
    duration = sum(segment.duration for segment in SEGMENTS)
    print(f"Generated {OUTPUT} ({duration // 60}:{duration % 60:02d})")


if __name__ == "__main__":
    main()
