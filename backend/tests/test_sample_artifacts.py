import unittest
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]


class SampleArtifactTests(unittest.TestCase):
    def test_readme_describes_real_retrieval_stack(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        forbidden = {
            "local hash-vector search",
            "Local hash-vector and BM25 retrieval",
            "Local hash vectors, BM25, deterministic reranker",
            r".venv\S cripts\a ctivate",
        }

        self.assertIn("SentenceTransformer embeddings with FAISS vector search", readme)
        self.assertIn(r".venv\Scripts\activate", readme)
        for phrase in forbidden:
            self.assertNotIn(phrase, readme)

    def test_nova_pdf_contains_chart_and_table_text(self):
        pdf = PdfReader(ROOT / "samples" / "documents" / "Nova_Retail_Annual_Report_2025.pdf")

        self.assertEqual(len(pdf.pages), 18)
        self.assertIn("Figure 3", pdf.pages[7].extract_text())
        self.assertIn("Table 2", pdf.pages[8].extract_text())
        self.assertIn("$128M", pdf.pages[8].extract_text())

    def test_simulated_samples_are_clearly_disclosed(self):
        log = (ROOT / "samples" / "logs" / "pcie_link_training.log").read_text(encoding="utf-8")
        issue = (ROOT / "samples" / "github-issues" / "issue-104-pcie-link-flap.json").read_text(encoding="utf-8")

        self.assertIn("Synthetic sample", log)
        self.assertIn("Synthetic GitHub issue", issue)


if __name__ == "__main__":
    unittest.main()
