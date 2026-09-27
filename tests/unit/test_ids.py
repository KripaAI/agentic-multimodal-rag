"""Identifier formats (LLD §3.1)."""

from __future__ import annotations

import re

import pytest

from mmrag.ingest.ids import content_hash, doc_id, element_id
from mmrag.ingest.parse import ParseContext

pytestmark = pytest.mark.unit


def test_doc_id_is_16_hex_of_file_bytes(tmp_path):
    a = tmp_path / "a.pdf"
    a.write_bytes(b"same bytes")
    b = tmp_path / "b.pdf"
    b.write_bytes(b"same bytes")
    c = tmp_path / "c.pdf"
    c.write_bytes(b"other bytes")
    assert re.fullmatch(r"[0-9a-f]{16}", doc_id(a))
    assert doc_id(a) == doc_id(b)  # depends on content, not the file name
    assert doc_id(a) != doc_id(c)


def test_element_id_format():
    assert element_id("0123456789abcdef", 12, "vector_figure", 3) == "0123456789abcdef:p12:vector_figure:3"


def test_content_hash_text_equals_utf8_bytes():
    h = content_hash("héllo")
    assert re.fullmatch(r"[0-9a-f]{64}", h)
    assert h == content_hash("héllo".encode("utf-8"))
    assert h != content_hash("hello")


def test_context_numbers_elements_per_page_and_type(parse_settings, tmp_path):
    ctx = ParseContext("d" * 16, "x.pdf", parse_settings.parse, tmp_path)
    assert ctx.next_element_id(1, "text") == "dddddddddddddddd:p1:text:1"
    assert ctx.next_element_id(1, "text") == "dddddddddddddddd:p1:text:2"
    assert ctx.next_element_id(1, "image") == "dddddddddddddddd:p1:image:1"
    assert ctx.next_element_id(2, "text") == "dddddddddddddddd:p2:text:1"
