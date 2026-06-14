export const demoDocuments = [
  {
    id: "nova-2025",
    name: "Nova Retail Annual Report 2025.pdf",
    pages: 18,
    chunks: 7,
    status: "ready",
    modalities: ["chart", "image", "scan", "table", "text"],
  },
  {
    id: "atlas-esg-2025",
    name: "Atlas ESG Review 2025.pdf",
    pages: 7,
    chunks: 2,
    status: "ready",
    modalities: ["chart", "table"],
  },
];

export const demoStats = {
  documents: 2,
  pages: 9,
  chunks: 9,
  modalities: { text: 2, table: 2, chart: 3, image: 1, scan: 1 },
  index_backend: "faiss:BAAI/bge-small-en-v1.5",
};

export const initialResult = {
  answer:
    "Revenue rises consistently from $78M in 2021 to $128M in 2025, a total increase of 64%. The largest annual gain occurs in 2025, when revenue grows by $20M. [1] Operating profit grows even faster, from $9M to $24M, expanding margin from 11.5% to 18.8%. [2]",
  retrieval_ms: 42,
  generation_ms: 318,
  model: "local-grounded-synthesizer",
  citations: [
    {
      index: 1,
      document_name: "Nova Retail Annual Report 2025.pdf",
      page: 8,
      modality: "chart",
      score: 0.932,
      excerpt:
        "Figure 3 is a line chart of annual revenue. Values rise from $78M in 2021, $86M in 2022, $97M in 2023, $108M in 2024, to $128M in 2025.",
    },
    {
      index: 2,
      document_name: "Nova Retail Annual Report 2025.pdf",
      page: 16,
      modality: "chart",
      score: 0.884,
      excerpt:
        "Sales increase from $78M in 2021 to $128M in 2025, while profit rises from $9M to $24M. Operating margin expands from 11.5% to 18.8%.",
    },
  ],
};
