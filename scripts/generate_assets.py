from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SHOTS = ROOT / "screenshots"
BG = "#08101b"
PANEL = "#101b29"
PANEL_2 = "#0c1521"
LINE = "#263345"
TEXT = "#e8edf5"
MUTED = "#7c899c"
LIME = "#b8ee45"
AQUA = "#68d5c9"
PURPLE = "#917cf6"


def font(size: int, bold: bool = False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def rounded(draw, box, radius=18, fill=PANEL, outline=LINE, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, size=24, color=TEXT, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=color, anchor=anchor)


def wrapped(draw, xy, value, width, size=22, color=MUTED, spacing=7):
    chars = max(12, int(width / (size * .55)))
    lines = wrap(value, chars)
    draw.multiline_text(xy, "\n".join(lines), font=font(size), fill=color, spacing=spacing)


def arrow(draw, start, end, color=LIME, width=4):
    draw.line([start, end], fill=color, width=width)
    x, y = end
    draw.polygon([(x, y), (x - 13, y - 8), (x - 13, y + 8)], fill=color)


def save_architecture():
    image = Image.new("RGB", (1800, 980), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (90, 70), "MULTIMODAL RAG ARCHITECTURE", 18, LIME, True)
    text(draw, (90, 105), "From unstructured documents to cited answers", 42, TEXT, True)
    stages = [
        ("01", "Sources", "PDFs, scans, images, tables, charts"),
        ("02", "Extraction", "pypdf text, pdfplumber tables, OCR and vision"),
        ("03", "Indexing", "500-token chunks, SentenceTransformer, FAISS"),
        ("04", "Retrieval", "Top-10 vector + BM25, CrossEncoder reranking"),
        ("05", "Response", "Local synthesis or optional Gemini, with citations"),
    ]
    x_positions = [75, 420, 765, 1110, 1455]
    for index, ((number, title, body), x) in enumerate(zip(stages, x_positions)):
        rounded(draw, (x, 300, x + 270, 660), 22)
        draw.ellipse((x + 24, 324, x + 72, 372), fill="#1f2d3d", outline=LIME, width=2)
        text(draw, (x + 48, 348), number, 15, LIME, True, "mm")
        text(draw, (x + 24, 405), title, 25, TEXT, True)
        wrapped(draw, (x + 24, 452), body, 220, 18, MUTED, 9)
        if index < len(stages) - 1:
            arrow(draw, (x + 281, 480), (x + 327, 480), "#55667a", 3)
    rounded(draw, (270, 755, 1530, 875), 18, "#0d1723")
    text(draw, (315, 785), "TRACEABILITY LAYER", 14, AQUA, True)
    text(draw, (315, 820), "Every chunk preserves document, page, chunk ID, modality, token range, and retrieval score.", 22, TEXT)
    DOCS.mkdir(exist_ok=True)
    image.save(DOCS / "architecture.png", quality=95)


def save_workflow():
    image = Image.new("RGB", (1800, 980), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (90, 70), "QUERY WORKFLOW", 18, LIME, True)
    text(draw, (90, 105), "Hybrid retrieval keeps exact values and semantic context", 40, TEXT, True)
    nodes = [
        (150, 270, "User question", "Compare Figure 3 and Table 2"),
        (590, 210, "Semantic path", "SentenceTransformer embeddings search FAISS"),
        (590, 530, "Lexical path", "BM25 preserves exact labels"),
        (1030, 370, "Fusion + reranking", "Top 10 candidates scored by CrossEncoder"),
        (1450, 370, "Grounded answer", "Citations link back to page evidence"),
    ]
    for x, y, title, body in nodes:
        rounded(draw, (x, y, x + 280, y + 180), 18)
        text(draw, (x + 24, y + 27), title, 21, TEXT, True)
        wrapped(draw, (x + 24, y + 72), body, 230, 17, MUTED, 8)
    arrow(draw, (430, 350), (575, 290), PURPLE, 4)
    arrow(draw, (430, 370), (575, 590), AQUA, 4)
    arrow(draw, (870, 290), (1015, 420), PURPLE, 4)
    arrow(draw, (870, 610), (1015, 480), AQUA, 4)
    arrow(draw, (1310, 460), (1435, 460), LIME, 4)
    rounded(draw, (590, 785, 1310, 875), 16, "#111c2a")
    text(draw, (950, 815), "Cross-encoder reranking reduces ten hybrid candidates to the strongest cited evidence", 19, LIME, False, "ma")
    image.save(DOCS / "workflow.png", quality=95)


def save_system_design():
    image = Image.new("RGB", (1800, 1080), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (90, 65), "SYSTEM DESIGN", 18, LIME, True)
    text(draw, (90, 100), "Implemented components in the local prototype", 42, TEXT, True)
    boxes = [
        (90, 260, 330, 390, "React dashboard", "Upload and query UI"),
        (450, 260, 690, 390, "FastAPI", "Document and query endpoints"),
        (830, 190, 1090, 320, "PDF extraction", "pypdf text + pdfplumber tables"),
        (830, 390, 1090, 520, "Image analysis", "Tesseract + optional Gemini Vision"),
        (1210, 290, 1480, 420, "Persistent index", "BGE vectors + FAISS + metadata"),
        (90, 700, 350, 830, "User question", "Natural-language query"),
        (470, 700, 730, 830, "Hybrid retrieval", "SentenceTransformer + FAISS + BM25"),
        (850, 700, 1110, 830, "Reranker", "SentenceTransformers CrossEncoder"),
        (1230, 700, 1530, 830, "Cited response", "Local synthesis or optional Gemini"),
    ]
    for x1, y1, x2, y2, title, body in boxes:
        rounded(draw, (x1, y1, x2, y2), 17)
        text(draw, ((x1 + x2) // 2, y1 + 45), title, 19, TEXT, True, "mm")
        text(draw, ((x1 + x2) // 2, y1 + 82), body, 13, MUTED, False, "mm")
    arrow(draw, (330, 325), (435, 325), "#526277", 3)
    arrow(draw, (690, 310), (815, 260), "#526277", 3)
    arrow(draw, (690, 340), (815, 450), "#526277", 3)
    arrow(draw, (1090, 255), (1195, 330), "#526277", 3)
    arrow(draw, (1090, 455), (1195, 380), "#526277", 3)
    draw.line((1345, 420, 1345, 590), fill="#526277", width=3)
    draw.line((1345, 590, 600, 590), fill="#526277", width=3)
    draw.line((600, 590, 600, 685), fill="#526277", width=3)
    draw.polygon([(600, 700), (593, 687), (607, 687)], fill="#526277")
    arrow(draw, (350, 765), (455, 765), LIME, 4)
    arrow(draw, (730, 765), (835, 765), LIME, 4)
    arrow(draw, (1110, 765), (1215, 765), LIME, 4)
    rounded(draw, (320, 930, 1480, 1010), 14, "#0d1723")
    text(draw, (900, 970), "Extracted chunks, embeddings, reranker scores, and the FAISS index are cached on disk.", 17, AQUA, False, "mm")
    image.save(DOCS / "system-design.png", quality=95)


def browser_frame(image, draw, title="Prism RAG"):
    draw.rectangle((0, 0, image.width, 58), fill="#121b28")
    for x, color in [(23, "#ff6b68"), (47, "#f6c85f"), (71, "#66c976")]:
        draw.ellipse((x, 22, x + 12, 34), fill=color)
    rounded(draw, (150, 14, image.width - 150, 44), 8, "#0b121c", "#253244")
    text(draw, (image.width // 2, 29), "localhost:5173  /  " + title, 12, MUTED, False, "mm")


def sidebar(draw, height):
    draw.rectangle((0, 58, 255, height), fill="#080f19")
    text(draw, (35, 93), "PRISM", 20, TEXT, True)
    text(draw, (35, 120), "MULTIMODAL RAG", 9, MUTED, True)
    items = [(190, "Workspace"), (235, "Documents     2"), (280, "Search history")]
    for y, label in items:
        if y == 190:
            rounded(draw, (22, y - 10, 233, y + 27), 7, "#172231", None)
        text(draw, (43, y), label, 13, TEXT if y == 190 else MUTED)
    text(draw, (34, 365), "INDEX HEALTH", 9, MUTED, True)
    draw.ellipse((34, 395, 43, 404), fill=LIME)
    text(draw, (53, 390), "Demo index ready", 12, TEXT, True)
    text(draw, (53, 410), "9 chunks indexed", 10, MUTED)
    rounded(draw, (22, 488, 233, 630), 10, PANEL, LINE)
    text(draw, (38, 510), "RETRIEVAL PIPELINE", 9, PURPLE, True)
    text(draw, (38, 548), "Hybrid + reranking", 13, TEXT, True)
    wrapped(draw, (38, 573), "Semantic and lexical evidence fused before generation.", 170, 10, MUTED, 4)


def save_home():
    image = Image.new("RGB", (1600, 1000), BG)
    draw = ImageDraw.Draw(image)
    browser_frame(image, draw)
    sidebar(draw, image.height)
    text(draw, (310, 100), "DOCUMENT INTELLIGENCE", 10, LIME, True)
    text(draw, (310, 125), "Ask across document evidence.", 27, TEXT, True)
    rounded(draw, (1290, 93, 1530, 135), 7, LIME, None)
    text(draw, (1410, 114), "Upload document", 13, "#09111c", True, "mm")
    rounded(draw, (305, 190, 1065, 575), 13)
    text(draw, (338, 220), "01", 12, LIME, True)
    text(draw, (382, 216), "QUERY WORKSPACE", 9, MUTED, True)
    text(draw, (382, 240), "What do you want to understand?", 21, TEXT, True)
    rounded(draw, (338, 292, 1032, 435), 9, "#09121d", "#415269")
    text(draw, (365, 325), "What is the revenue trend shown in Figure 3?", 18, TEXT)
    draw.line((338, 390, 1032, 390), fill=LINE, width=1)
    text(draw, (365, 407), "Grounded in 2 documents", 10, MUTED)
    rounded(draw, (976, 398, 1017, 426), 6, LIME, None)
    text(draw, (338, 468), "TRY AN EXAMPLE", 9, MUTED, True)
    for i, label in enumerate(["Compare Figure 3 and Table 2", "Summarize pages 12-18", "How are sales and profit related?"]):
        x = 338 + (i % 2) * 345
        y = 495 + (i // 2) * 48
        rounded(draw, (x, y, x + 326, y + 36), 6, "#0c1622", LINE)
        text(draw, (x + 13, y + 10), label, 10, MUTED)
    rounded(draw, (1088, 190, 1530, 575), 13)
    text(draw, (1118, 220), "INDEX OVERVIEW", 9, MUTED, True)
    text(draw, (1118, 272), "09", 55, TEXT)
    text(draw, (1190, 300), "PAGES INDEXED", 9, MUTED)
    draw.line((1118, 350, 1500, 350), fill=LINE, width=1)
    text(draw, (1118, 380), "2", 25, TEXT, True)
    text(draw, (1118, 415), "DOCUMENTS", 9, MUTED, True)
    text(draw, (1320, 380), "9", 25, TEXT, True)
    text(draw, (1320, 415), "MODAL SIGNALS", 9, MUTED, True)
    rounded(draw, (305, 610, 1530, 940), 13)
    text(draw, (338, 640), "02", 12, LIME, True)
    text(draw, (382, 636), "GROUNDED RESPONSE", 9, MUTED, True)
    text(draw, (382, 660), "Evidence, synthesized.", 21, TEXT, True)
    wrapped(draw, (338, 725), "Revenue rises consistently from $78M in 2021 to $128M in 2025. The largest annual gain occurs in 2025, while operating profit grows faster than sales.", 700, 19, TEXT, 9)
    rounded(draw, (1110, 700, 1495, 900), 10, "#0c1622", LINE)
    text(draw, (1135, 725), "RETRIEVED EVIDENCE", 9, MUTED, True)
    text(draw, (1135, 770), "[1]  Page 8  /  chart", 12, LIME, True)
    text(draw, (1135, 815), "[2]  Page 16  /  chart", 12, AQUA, True)
    SHOTS.mkdir(exist_ok=True)
    image.save(SHOTS / "home-page.png", quality=95)


def save_query():
    image = Image.open(SHOTS / "home-page.png").copy()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((334, 288, 1036, 439), radius=11, outline=LIME, width=3)
    rounded(draw, (540, 455, 845, 489), 8, "#1b2635", "#48586d")
    text(draw, (692, 472), "Hybrid search running: BGE + FAISS + BM25", 11, TEXT, False, "mm")
    image.save(SHOTS / "query-example.png", quality=95)


def save_result():
    image = Image.open(SHOTS / "home-page.png").copy()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((300, 605, 1535, 945), radius=15, outline=LIME, width=3)
    rounded(draw, (935, 630, 1105, 657), 7, "#17271f", "#3b5930")
    text(draw, (1020, 643), "2 SOURCES ATTACHED", 8, LIME, True, "mm")
    image.save(SHOTS / "result-example.png", quality=95)


def main():
    save_architecture()
    save_workflow()
    save_system_design()
    save_home()
    save_query()
    save_result()
    print("Generated documentation diagrams and product screenshots.")


if __name__ == "__main__":
    main()
