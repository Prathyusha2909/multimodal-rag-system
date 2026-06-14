from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "samples" / "documents"
PAGE_WIDTH, PAGE_HEIGHT = letter


def header(pdf: canvas.Canvas, title: str, page: int) -> None:
    pdf.setFillColor(colors.HexColor("#0f1b2a"))
    pdf.rect(0, PAGE_HEIGHT - 70, PAGE_WIDTH, 70, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#b8ee45"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(42, PAGE_HEIGHT - 27, "SYNTHETIC SAMPLE REPORT")
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(42, PAGE_HEIGHT - 50, title)
    pdf.setFillColor(colors.HexColor("#667386"))
    pdf.setFont("Helvetica", 8)
    pdf.drawRightString(PAGE_WIDTH - 42, 28, f"Page {page}")


def paragraph(pdf: canvas.Canvas, text: str, y: float, width: float = 510) -> None:
    style = getSampleStyleSheet()["BodyText"]
    style.fontName = "Helvetica"
    style.fontSize = 10
    style.leading = 15
    style.textColor = colors.HexColor("#263445")
    block = Paragraph(text, style)
    _, height = block.wrap(width, 300)
    block.drawOn(pdf, 48, y - height)


def draw_revenue_chart(pdf: canvas.Canvas) -> None:
    values = [78, 86, 97, 108, 128]
    years = [2021, 2022, 2023, 2024, 2025]
    left, bottom, width, height = 80, 245, 440, 270
    pdf.setStrokeColor(colors.HexColor("#b7c1cf"))
    pdf.line(left, bottom, left, bottom + height)
    pdf.line(left, bottom, left + width, bottom)
    points = []
    for index, (year, value) in enumerate(zip(years, values)):
        x = left + index * (width / 4)
        y = bottom + (value - 65) / 70 * height
        points.append((x, y))
        pdf.setFillColor(colors.HexColor("#536174"))
        pdf.setFont("Helvetica", 8)
        pdf.drawCentredString(x, bottom - 18, str(year))
        pdf.setFillColor(colors.HexColor("#0f1b2a"))
        pdf.drawCentredString(x, y + 12, f"${value}M")
    pdf.setStrokeColor(colors.HexColor("#78a817"))
    pdf.setLineWidth(3)
    for first, second in zip(points, points[1:]):
        pdf.line(first[0], first[1], second[0], second[1])
    for x, y in points:
        pdf.setFillColor(colors.HexColor("#b8ee45"))
        pdf.circle(x, y, 5, fill=1, stroke=0)


def build_nova() -> None:
    path = OUTPUT / "Nova_Retail_Annual_Report_2025.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    page_copy = {
        3: ("Executive Summary", "Nova Retail delivered resilient growth in 2025. Revenue reached $128 million, up 18% year over year, while operating profit increased to $24 million."),
        12: ("Customer Growth", "Active customers grew 14% to 3.2 million. Mobile accounted for 61% of online orders, and repeat purchase frequency improved from 3.1 to 3.6 orders per year."),
        14: ("Distribution Network", "New distribution hubs opened in Pune, Hyderabad, and Chennai. The expanded network places 82% of customers within one-day delivery range."),
        16: ("Sales and Operating Profit", "Sales increased from $78M in 2021 to $128M in 2025, while operating profit rose from $9M to $24M. Operating margin expanded from 11.5% to 18.8%."),
        18: ("Risk Committee Note", "Supply-chain concentration remains the principal operational risk. Two suppliers were added for critical private-label goods and safety stock was increased for the top 50 products."),
    }
    for page in range(1, 19):
        header(pdf, "Nova Retail Annual Report 2025", page)
        if page == 8:
            pdf.setFillColor(colors.HexColor("#0f1b2a"))
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(48, 670, "Figure 3 - Annual Revenue Trend")
            draw_revenue_chart(pdf)
            paragraph(pdf, "Revenue increased in each reported year, with the largest annual increase of $20M occurring in 2025.", 205)
        elif page == 9:
            pdf.setFillColor(colors.HexColor("#0f1b2a"))
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(48, 670, "Table 2 - Segment Performance")
            data = [
                ["Segment", "Revenue", "Profit", "Margin"],
                ["Online", "$62M", "$15M", "24.2%"],
                ["Stores", "$51M", "$8M", "15.7%"],
                ["Wholesale", "$15M", "$1M", "6.7%"],
                ["Total", "$128M", "$24M", "18.8%"],
            ]
            table = Table(data, colWidths=[2.2 * inch, 1.25 * inch, 1.25 * inch, 1.1 * inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f1b2a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eef3f7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a9b5c3")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("PADDING", (0, 0), (-1, -1), 10),
            ]))
            table.wrapOn(pdf, 500, 500)
            table.drawOn(pdf, 48, 420)
        else:
            title, body = page_copy.get(page, (f"Section {page}", "This synthetic page is included to provide realistic pagination for upload and citation testing."))
            pdf.setFillColor(colors.HexColor("#0f1b2a"))
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(48, 670, title)
            paragraph(pdf, body, 635)
        pdf.showPage()
    pdf.save()


def build_atlas() -> None:
    path = OUTPUT / "Atlas_ESG_Review_2025.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    for page in range(1, 8):
        header(pdf, "Atlas ESG Review 2025", page)
        if page == 4:
            pdf.setFillColor(colors.HexColor("#0f1b2a"))
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(48, 670, "Emissions Scorecard")
            paragraph(pdf, "Scope 1 emissions were 8,200 tCO2e, Scope 2 emissions were 12,400 tCO2e, and Scope 3 emissions were 96,000 tCO2e. Scope 3 represented 82% of the measured footprint.", 630)
        elif page == 7:
            pdf.setFillColor(colors.HexColor("#0f1b2a"))
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(48, 670, "Renewable Electricity Adoption")
            paragraph(pdf, "Renewable electricity increased from 24% in 2021 to 68% in 2025. Atlas targets 90% renewable electricity by 2028.", 630)
        else:
            pdf.setFillColor(colors.HexColor("#0f1b2a"))
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(48, 670, f"ESG Section {page}")
            paragraph(pdf, "This synthetic page supports document upload, parsing, and page-level citation examples.", 630)
        pdf.showPage()
    pdf.save()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_nova()
    build_atlas()
    print(f"Generated sample PDFs in {OUTPUT}")


if __name__ == "__main__":
    main()

