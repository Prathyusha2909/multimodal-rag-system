from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "demo" / "demo.gif"
SIZE = (960, 600)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def screenshot(name: str) -> Image.Image:
    return Image.open(ROOT / "screenshots" / name).convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)


def title_frame(title: str, subtitle: str) -> Image.Image:
    frame = Image.new("RGB", SIZE, "#07111d")
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle((70, 92, 260, 100), radius=4, fill="#b8ee45")
    draw.text((70, 155), title, font=font(48, True), fill="#f2f6fb")
    draw.text((72, 235), subtitle, font=font(24), fill="#b8c8d9")
    draw.text((72, 505), "multimodal-rag-system", font=font(20, True), fill="#b8ee45")
    return frame


def annotated_frame(name: str, caption: str, box: tuple[int, int, int, int]) -> Image.Image:
    frame = screenshot(name)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rectangle((0, 0, SIZE[0], 66), fill=(5, 12, 20, 235))
    draw.text((30, 18), caption, font=font(27, True), fill="#f2f6fb")
    scaled = tuple(round(value * 0.6) for value in box)
    for offset in range(4):
        draw.rounded_rectangle(
            (scaled[0] - offset, scaled[1] - offset, scaled[2] + offset, scaled[3] + offset),
            radius=10,
            outline=(184, 238, 69, 245 - offset * 35),
            width=3,
        )
    return frame


def evidence_frame() -> Image.Image:
    frame = Image.new("RGB", SIZE, "#07111d")
    draw = ImageDraw.Draw(frame)
    draw.text((45, 34), "6. Open the retrieved chunk behind each citation", font=font(30, True), fill="#f2f6fb")
    draw.rounded_rectangle((45, 110, 915, 535), radius=18, fill="#111f2e", outline="#536f86", width=2)
    draw.text((78, 145), "RETRIEVED EVIDENCE  [1]", font=font(18, True), fill="#b8ee45")
    draw.text((78, 193), "Nova Retail Annual Report 2025.pdf", font=font(27, True), fill="#f2f6fb")
    draw.text((78, 238), "Page 8  |  chart  |  reranked result", font=font(20), fill="#67d8d1")
    lines = [
        "Figure 3 is a line chart of annual revenue. Values rise from",
        "$78M in 2021, $86M in 2022, $97M in 2023, $108M in 2024,",
        "to $128M in 2025.",
    ]
    draw.multiline_text((78, 310), "\n".join(lines), font=font(24), fill="#dbe7f2", spacing=14)
    draw.text((78, 485), "Inspectable evidence, not a hidden confidence claim", font=font(18, True), fill="#9eb0c2")
    return frame


def main() -> None:
    frames = [
        title_frame("Multimodal RAG System", "From mixed documents to inspectable citations"),
        annotated_frame("home-page.png", "1. Upload a PDF, scan, image, text file, or CSV", (1284, 88, 1535, 142)),
        annotated_frame("query-example.png", "2. Ask a question across text, tables, charts, and scans", (333, 286, 1040, 442)),
        annotated_frame("query-example.png", "3. Retrieve with BGE + FAISS + BM25, then rerank", (17, 482, 240, 637)),
        annotated_frame("result-example.png", "4. Generate a grounded answer from the best evidence", (298, 602, 1105, 948)),
        annotated_frame("result-example.png", "5. Inspect cited pages, modalities, and retrieved chunks", (1105, 695, 1502, 908)),
        evidence_frame(),
        title_frame("Try the live demo", "multimodal-rag-system-pink.vercel.app"),
    ]
    durations = [4000, 7000, 7000, 7000, 7000, 5000, 5000, 4000]
    OUTPUT.parent.mkdir(exist_ok=True)
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Generated {OUTPUT} ({sum(durations) // 1000} seconds)")


if __name__ == "__main__":
    main()
