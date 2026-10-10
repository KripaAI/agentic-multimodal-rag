"""JSONL reading splits on newlines only (Phase 10: parashara-light-manual-6.pdf could not be
indexed because a text block contained \\x85, which str.splitlines() treats as a line break)."""

from __future__ import annotations

import pytest

from mmrag.fsutil import jsonl_lines
from mmrag.ingest.models import Element

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("odd", ["\x85", " ", " ", "\x0b", "\x0c", "\x1c"])
def test_a_record_with_a_unicode_line_break_stays_whole(tmp_path, odd):
    e = Element(element_id="d:p1:text:1", doc_id="d", source_file="x.pdf", page=1, bbox=(0, 0, 1, 1), type="text",
                text=f"Select the lord{odd}of the 9th", content_hash="h")
    path = tmp_path / "elements.jsonl"
    path.write_text(e.model_dump_json() + "\n" + e.model_dump_json() + "\n\n", encoding="utf-8")
    lines = jsonl_lines(path)
    assert len(lines) == 2 and Element.model_validate_json(lines[0]).text == e.text


def test_long_hindi_text_is_cut_between_characters():
    """A transcribed Lal Kitab page is longer than one chunk: the cut must not split a character."""
    from mmrag.index.chunk import _pieces

    text = "शनि ख़ाना नं० 10 में हो तो नेक फल होगा। " * 120
    e = Element(element_id="d:p1:scanned_page:1", doc_id="d", source_file="x.pdf", page=1, bbox=(0, 0, 1, 1),
                type="scanned_page", text=text, content_hash="h")
    pieces = _pieces(e, 100)
    assert len(pieces) > 1 and "".join(pieces) == text and not any("\ufffd" in p for p in pieces)
