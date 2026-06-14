from app.domain import DocumentChunk


def build_sample_corpus() -> list[DocumentChunk]:
    """A compact synthetic report used by the no-setup product demo."""
    return [
        DocumentChunk(
            id="nova-03-text",
            document_id="nova-2025",
            document_name="Nova Retail Annual Report 2025.pdf",
            page=3,
            modality="text",
            title="Executive summary",
            content=(
                "Nova Retail delivered resilient growth in 2025. Revenue reached $128 million, "
                "up 18% year over year, while operating profit increased to $24 million. "
                "Management attributes the improvement to digital sales and better inventory turns."
            ),
        ),
        DocumentChunk(
            id="nova-08-chart",
            document_id="nova-2025",
            document_name="Nova Retail Annual Report 2025.pdf",
            page=8,
            modality="chart",
            title="Figure 3 - Revenue trend",
            content=(
                "Figure 3 is a line chart of annual revenue. Values rise from $78M in 2021, "
                "$86M in 2022, $97M in 2023, $108M in 2024, to $128M in 2025. The trend is "
                "consistently upward, with the largest increase of $20M occurring in 2025."
            ),
            metadata={"figure": 3},
        ),
        DocumentChunk(
            id="nova-09-table",
            document_id="nova-2025",
            document_name="Nova Retail Annual Report 2025.pdf",
            page=9,
            modality="table",
            title="Table 2 - Segment performance",
            content=(
                "Table 2 lists segment performance for 2025. Online: revenue $62M, profit $15M, "
                "margin 24.2%. Stores: revenue $51M, profit $8M, margin 15.7%. Wholesale: revenue "
                "$15M, profit $1M, margin 6.7%. Total revenue is $128M and total profit is $24M."
            ),
            metadata={"table": 2},
        ),
        DocumentChunk(
            id="nova-12-text",
            document_id="nova-2025",
            document_name="Nova Retail Annual Report 2025.pdf",
            page=12,
            modality="text",
            title="Customer growth",
            content=(
                "Active customers grew 14% to 3.2 million. Mobile accounted for 61% of online "
                "orders, and repeat purchase frequency improved from 3.1 to 3.6 orders per year."
            ),
        ),
        DocumentChunk(
            id="nova-14-image",
            document_id="nova-2025",
            document_name="Nova Retail Annual Report 2025.pdf",
            page=14,
            modality="image",
            title="Distribution network map",
            content=(
                "A map shows three new distribution hubs in Pune, Hyderabad, and Chennai. "
                "The expanded network places 82% of customers within one-day delivery range."
            ),
        ),
        DocumentChunk(
            id="nova-16-chart",
            document_id="nova-2025",
            document_name="Nova Retail Annual Report 2025.pdf",
            page=16,
            modality="chart",
            title="Sales and operating profit",
            content=(
                "A grouped bar and line graph compares sales and operating profit. Sales increase "
                "from $78M in 2021 to $128M in 2025, while profit rises from $9M to $24M. Profit "
                "grows faster than sales, so operating margin expands from 11.5% to 18.8%."
            ),
        ),
        DocumentChunk(
            id="nova-18-scan",
            document_id="nova-2025",
            document_name="Nova Retail Annual Report 2025.pdf",
            page=18,
            modality="scan",
            title="Risk committee note",
            content=(
                "OCR from a scanned risk note: supply-chain concentration remains the principal "
                "operational risk. The company added two suppliers for critical private-label goods "
                "and increased safety stock for the top 50 products."
            ),
        ),
        DocumentChunk(
            id="atlas-04-table",
            document_id="atlas-esg-2025",
            document_name="Atlas ESG Review 2025.pdf",
            page=4,
            modality="table",
            title="Emissions scorecard",
            content=(
                "The emissions table reports Scope 1 at 8,200 tCO2e, Scope 2 at 12,400 tCO2e, "
                "and Scope 3 at 96,000 tCO2e. Scope 3 is 82% of the measured footprint."
            ),
        ),
        DocumentChunk(
            id="atlas-07-chart",
            document_id="atlas-esg-2025",
            document_name="Atlas ESG Review 2025.pdf",
            page=7,
            modality="chart",
            title="Renewable energy adoption",
            content=(
                "The chart shows renewable electricity increasing from 24% in 2021 to 68% in 2025. "
                "The company targets 90% renewable electricity by 2028."
            ),
        ),
    ]

