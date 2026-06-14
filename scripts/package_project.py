from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "multimodal-rag-system.zip"
ARCHIVE_ROOT = "multimodal-rag-system"

EXCLUDED_DIRS = {
    ".git",
    ".deepeval",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
}
EXCLUDED_FILES = {
    ".coverage",
    ".env",
    OUTPUT.name,
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log"}
EXCLUDED_PATH_PREFIXES = {
    ("backend", "data", "cache"),
    ("backend", "data", "index"),
}


def should_include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(relative.parts[: len(prefix)] == prefix for prefix in EXCLUDED_PATH_PREFIXES):
        return False
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    is_published_sample_log = relative.parts[:2] == ("samples", "logs") and path.suffix == ".log"
    if path.name in EXCLUDED_FILES or (path.suffix in EXCLUDED_SUFFIXES and not is_published_sample_log):
        return False
    if relative.parts[:3] == ("backend", "data", "uploads") and path.name != ".gitkeep":
        return False
    return True


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and should_include(path))
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path(ARCHIVE_ROOT) / path.relative_to(ROOT))
    print(f"Created {OUTPUT} with {len(files)} files ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
