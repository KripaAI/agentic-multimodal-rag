"""Search documents for the three collections (LLD §3.6, spec §6.4).

- text: section-aware chunks of `min_tokens`–`max_tokens`, with about `overlap` carried
  over as whole paragraphs; never crossing a section; prefixed with the section path.
- figure: caption + description + labels + keywords + PDF caption + linked paragraphs.
- table: meaning (summary, title, columns) and keywords (title, columns, every cell) differ.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import groupby
from typing import Literal

import tiktoken
from pydantic import BaseModel, ConfigDict

from mmrag.config import Chunk as ChunkConfig
from mmrag.index.document import Link, LoadedDoc
from mmrag.ingest.models import Element

ENCODING = "cl100k_base"  # the tokenizer of OpenAI's text-embedding-3 models


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    collection: Literal["text", "figure", "table"]
    element_ids: list[str]  # reading order
    dense_text: str  # embedded
    keyword_text: str  # full-text indexed


@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def _pieces(block: Element, limit: int) -> list[str]:
    """A paragraph, or token slices of it when it alone exceeds the limit."""
    tokens = _encoder().encode(block.text)
    if len(tokens) <= limit:
        return [block.text]
    return [_encoder().decode(tokens[i:i + limit]) for i in range(0, len(tokens), limit)]


def chunk_text(doc: LoadedDoc, cfg: ChunkConfig) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(label: str, parts: list[tuple[str, str]]) -> None:
        body = "\n\n".join(text for _, text in parts)
        ids = list(dict.fromkeys(eid for eid, _ in parts))
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}:text:{len(chunks) + 1}", collection="text", element_ids=ids,
                            dense_text=f"{label}\n{body}", keyword_text=f"{label}\n{body}"))

    overlap_budget = int(cfg.max_tokens * cfg.overlap)
    for section, blocks in groupby(doc.texts(), key=lambda e: tuple(e.section_path)):
        label = " > ".join(section) or "(no section)"
        budget = cfg.max_tokens - count_tokens(label) - 1
        parts: list[tuple[str, str, int]] = []  # (element_id, text, tokens)
        for block in blocks:
            for piece in _pieces(block, budget):
                n = count_tokens(piece)
                if parts and sum(p[2] for p in parts) + n > budget:
                    emit(label, [(e, t) for e, t, _ in parts])
                    carry, used = [], 0  # whole trailing paragraphs, up to the overlap budget
                    for p in reversed(parts):
                        if used + p[2] > overlap_budget:
                            break
                        carry.insert(0, p)
                        used += p[2]
                    parts = carry if used + n <= budget else []
                parts.append((block.element_id, piece, n))
        if parts:
            emit(label, [(e, t) for e, t, _ in parts])
    return chunks


def build_figure_docs(doc: LoadedDoc, links: list[Link]) -> list[Chunk]:
    texts = {e.element_id: e for e in doc.texts()}
    linked: dict[str, list[str]] = {}
    for link in links:
        if link.text_element_id in texts:
            linked.setdefault(link.target_id, []).append(texts[link.text_element_id].text)

    chunks = []
    for n, fig in enumerate(doc.figures(), 1):
        parts = []
        record = doc.captions.get(fig.element_id)
        if record and record.caption:
            c = record.caption
            parts += [c.short_caption, c.detailed_description]
            if c.visible_text:
                parts.append("Labels: " + " | ".join(c.visible_text))
            if c.keywords:
                parts.append("Keywords: " + ", ".join(c.keywords))
        elif fig.text:  # not captioned: the PDF's own labels still make it findable
            parts.append("Labels: " + " | ".join(fig.text.splitlines()))
        if fig.caption:
            parts.append(f"Caption: {fig.caption}")
        if fig.section_path:
            parts.append("Section: " + " > ".join(fig.section_path))
        parts += [f"Context: {t}" for t in linked.get(fig.element_id, [])]
        text = "\n".join(parts)
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}:figure:{n}", collection="figure", element_ids=[fig.element_id],
                            dense_text=text, keyword_text=text))
    return chunks


def build_table_docs(doc: LoadedDoc) -> list[Chunk]:
    chunks = []
    for n, el in enumerate(doc.table_elements(), 1):
        t = doc.tables[el.element_id]
        head = [t.title or "", "Columns: " + " | ".join(t.columns)]
        dense = "\n".join([*head[:1], t.summary or "", *head[1:], "Section: " + " > ".join(el.section_path)])
        cells = "\n".join(" | ".join(row) for row in t.rows)
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}:table:{n}", collection="table", element_ids=[el.element_id],
                            dense_text=dense.strip(), keyword_text="\n".join([*head, cells]).strip()))
    return chunks
