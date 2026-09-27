"""Review sheet: one self-contained HTML page for the owner to check detection (plan Phase 1, task 7).

Per page: a thumbnail with every detected figure, table and rejected region
outlined, followed by the cropped figure/table PNGs and the rejected regions
with the filter that dropped them. Summary counts at the top.
"""

from __future__ import annotations

from pathlib import Path

from mmrag.config import Settings
from mmrag.ingest.models import ParseResult


def build_review_sheet(result: ParseResult, pdf_path: Path, settings: Settings, out: Path) -> Path:
    """Write the review sheet to `out`; images are referenced by relative path."""
    raise NotImplementedError
