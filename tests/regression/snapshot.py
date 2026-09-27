"""Frozen, owner-approved parse results (plan Phase 1 task 10, constitution W5).

A snapshot records what detection found in a whole PDF: element counts, and the page,
box and caption of every figure, image and table. `test_parse_regression.py` re-parses
the PDF and compares. After the owner approves a detection change on the review sheet,
refresh the snapshots from the project root:

    python -m tests.regression.snapshot
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from mmrag.ingest.models import ParseResult

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
# The approved documents (file name in data/pdfs -> snapshot name).
DOCUMENTS = {
    "Transformers-in-Practice-Illustrated.pdf": "transformers",
    "Buildig-multimodal-rag.pdf": "buildig",
}


def snapshot(result: ParseResult) -> dict:
    """The approved facts about one parse, in a stable, reviewable form."""
    tables = {t.element_id: t for t in result.tables}
    items = []
    for e in result.elements:
        if e.type == "text" or e.status != "ok":
            continue
        item = {"page": e.page, "type": e.type, "bbox": [round(v, 1) for v in e.bbox]}
        if e.caption:
            item["caption"] = e.caption[:60]
        if e.type == "table":
            item["columns"] = tables[e.element_id].columns
        items.append(item)
    return {
        "doc_id": result.doc_id,
        "source_file": result.source_file,
        "pages": result.page_count,
        "counts": dict(sorted(Counter(f"{e.type}:{e.status}" for e in result.elements).items())),
        "skips": dict(sorted(Counter(s.kind for s in result.skips).items())),
        "items": items,
    }


def main() -> None:
    from mmrag.config import get_settings
    from mmrag.ingest.parse import parse_document

    settings = get_settings()
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    for pdf, name in DOCUMENTS.items():
        result = parse_document(settings.resolve(settings.paths.pdf_dir) / pdf, settings)
        path = SNAPSHOT_DIR / f"{name}.json"
        path.write_text(json.dumps(snapshot(result), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{path.name}: {len(snapshot(result)['items'])} items")


if __name__ == "__main__":
    main()
