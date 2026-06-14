# Sample Data

All files in this directory are synthetic and safe to publish. They are included so the project can be reviewed without private company documents, internal servers, or physical test equipment.

## Contents

- `documents/`: generated annual-report PDFs containing text, a table, and charts
- `logs/`: simulated PCIe and firmware-style validation logs
- `github-issues/`: simulated issue records in GitHub-compatible JSON form
- `outputs/`: example API responses and reproducible evaluation results

The PDF reports correspond to the curated typed chunks in `backend/app/services/sample_corpus.py`. Uploaded PDFs are parsed for extractable text; the demo corpus adds explicit modality labels and visual descriptions so chart/table retrieval can be tested deterministically.

Regenerate the PDFs with:

```bash
python scripts/generate_sample_data.py
```

