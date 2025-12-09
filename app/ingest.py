from pathlib import Path
import json
from typing import List, Dict, Any

# Project structure:
# physical-ai-book/
#   ├─ book-app/docs/          (textbook)
#   └─ agent-backend/app/ingest.py  (this file)

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # physical-ai-book
BACKEND_DIR = PROJECT_ROOT / "agent-backend"

DOCS_SOURCE_DIR = PROJECT_ROOT / "book-app" / "docs"
INDEX_FILE = BACKEND_DIR / "data" / "index.json"


def read_markdown_file(file_path: Path) -> Dict[str, Any]:
    """
    Read a markdown file and return id, relative path, and full text.
    """
    text_content = file_path.read_text(encoding="utf-8")

    relative_path = file_path.relative_to(DOCS_SOURCE_DIR).as_posix()
    doc_id = relative_path.replace(".md", "").replace("/", "_")

    return {
        "id": doc_id,
        "path": relative_path,
        "text": text_content,
    }


def ingest_book() -> Dict[str, Any]:
    """
    Recursively loads all .md files from DOCS_SOURCE_DIR and stores
    their content and relative paths into INDEX_FILE.
    """
    documents: List[Dict[str, Any]] = []

    if not DOCS_SOURCE_DIR.exists():
        return {
            "status": "error",
            "message": f"Documentation source directory not found: {DOCS_SOURCE_DIR}",
        }

    for file_path in DOCS_SOURCE_DIR.rglob("*.md"):
        if file_path.is_file():
            try:
                doc_data = read_markdown_file(file_path)
                documents.append(doc_data)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "indexed_documents": len(documents),
        "index_path": str(INDEX_FILE),
    }


if __name__ == "__main__":
    result = ingest_book()
    print(f"Ingestion Result: {result}")
