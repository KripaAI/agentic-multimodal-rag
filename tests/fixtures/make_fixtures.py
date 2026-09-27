"""Cut the fixture pages out of the source PDFs (LLD §10).

Run once from the project root: `python tests/fixtures/make_fixtures.py`.
The expected results next to each page (`*.json`) are written by hand.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
PDFS = ROOT / "data" / "pdfs"
OUT = Path(__file__).resolve().parent / "pages"

PAGES = {
    "transformers_p003": ("Transformers-in-Practice-Illustrated.pdf", 3),
    "transformers_p005": ("Transformers-in-Practice-Illustrated.pdf", 5),
    "transformers_p009": ("Transformers-in-Practice-Illustrated.pdf", 9),
    "transformers_p011": ("Transformers-in-Practice-Illustrated.pdf", 11),
    "transformers_p016": ("Transformers-in-Practice-Illustrated.pdf", 16),
    "transformers_p023": ("Transformers-in-Practice-Illustrated.pdf", 23),
    "transformers_p029": ("Transformers-in-Practice-Illustrated.pdf", 29),
    "transformers_p031": ("Transformers-in-Practice-Illustrated.pdf", 31),
    "buildig_p029": ("Buildig-multimodal-rag.pdf", 29),
    "buildig_p074": ("Buildig-multimodal-rag.pdf", 74),
    "llm_notes_p007": ("LLM_Training_and_Model_Lifecycle_Notes (2).pdf", 7),
    "llm_notes_p054": ("LLM_Training_and_Model_Lifecycle_Notes (2).pdf", 54),
    "llm_notes_p055": ("LLM_Training_and_Model_Lifecycle_Notes (2).pdf", 55),
    "buildig_p056": ("Buildig-multimodal-rag.pdf", 56),
    "post_training_p022": ("The_Complete_Guide_to_Post_Training_LLMs_v2_Expert_Edition.pdf", 22),
    "pre_training_p001": ("The_Complete_Guide_to_Pre_Training_LLMs_v2_Expert_Edition.pdf", 1),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (pdf, page) in PAGES.items():
        if (OUT / f"{name}.pdf").exists():
            continue  # re-cutting changes the file bytes (timestamps) with no content change
        src = pymupdf.open(PDFS / pdf)
        dst = pymupdf.open()
        dst.insert_pdf(src, from_page=page - 1, to_page=page - 1)
        dst.save(OUT / f"{name}.pdf", garbage=4, deflate=True)
        print(f"{name}.pdf  <- {pdf} p.{page}")


if __name__ == "__main__":
    main()
