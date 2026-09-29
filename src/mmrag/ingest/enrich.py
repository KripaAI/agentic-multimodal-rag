"""Enrichment between captioning and indexing (LLD §3.5, spec §6.3).

- `cross_check_labels`: the VLM's visible text must match the PDF's own labels.
- `find_links`: which paragraph discusses each figure and table. Rules in order: explicit
  "Figure/Table N", then a deictic phrase ("the diagram below"), then the *most related*
  nearby paragraph by meaning and shared labels (not simply the nearest one).
- `summarize_tables`: a two-sentence summary per table, cached per (table, model).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Callable

from mmrag.config import Enrich
from mmrag.index.document import Link, LoadedDoc
from mmrag.ingest.models import Element, Table

EmbedFn = Callable[[list[str]], list[list[float]]]
ChatFn = Callable[[str, str], str]  # (model, prompt) -> reply

_WORD = re.compile(r"[a-z0-9]+")
_FIGURE_NUMBER = re.compile(r"^\s*(?:figure|fig\.)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_TABLE_NUMBER = re.compile(r"^\s*table\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_VISUAL = r"(?:diagram|figure|chart|table|graph|picture|illustration|image)"
_DEICTIC_BELOW = re.compile(rf"\b(?:{_VISUAL}\s+below|(?:shown|illustrated|seen|pictured)\s+below|"
                            rf"(?:the\s+)?following\s+{_VISUAL}|below\s*:)", re.IGNORECASE)
_DEICTIC_ABOVE = re.compile(rf"\b(?:{_VISUAL}\s+above|(?:shown|illustrated|seen|pictured)\s+above|"
                            rf"(?:the\s+)?(?:previous|preceding)\s+{_VISUAL})", re.IGNORECASE)


MIN_CANDIDATE_WORDS = 8  # shorter blocks are headings or labels, not discussion


def _norm(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def _fold(text: str) -> str:
    """NFKC folds subscripts and compatibility forms: the PDF's q₁k₁ matches the model's q1k1."""
    return unicodedata.normalize("NFKC", text).lower()


def _words(text: str) -> set[str]:
    return set(_WORD.findall(_fold(text)))


# ---------------------------------------------------------------- label cross-check

def cross_check_labels(doc: LoadedDoc, cfg: Enrich) -> dict[str, float]:
    """For drawn figures, the share of the VLM's visible-text words that also appear in the
    PDF's own labels. Below `label_overlap_min`, the caption is marked needs_review (spec §6.3).
    Returns the share per checked figure."""
    shares = {}
    for fig in doc.figures():
        record = doc.captions.get(fig.element_id)
        if fig.type != "vector_figure" or not fig.text or not record or not record.caption:
            continue
        seen = _words(" ".join(record.caption.visible_text))
        if not seen:
            continue
        share = len(seen & _words(fig.text)) / len(seen)
        shares[fig.element_id] = round(share, 3)
        if share < cfg.label_overlap_min and record.status == "ok":
            doc.captions[fig.element_id] = record.model_copy(update={
                "status": "needs_review",
                "error": f"labels: only {share:.0%} of the words the model read appear in the PDF's labels",
            })
    return shares


# ---------------------------------------------------------------- links

def _target_label(target: Element, doc: LoadedDoc) -> str | None:
    return doc.tables[target.element_id].title if target.type == "table" else target.caption


def _target_text(target: Element, doc: LoadedDoc) -> str:
    """What a figure or table is about, for the meaning comparison."""
    if target.type == "table":
        t = doc.tables[target.element_id]
        return "\n".join(x for x in [t.title or "", t.summary or "", " | ".join(t.columns)] if x)
    parts = []
    record = doc.captions.get(target.element_id)
    if record and record.caption:
        parts += [record.caption.short_caption, record.caption.detailed_description]
    if target.caption:
        parts.append(target.caption)
    if not parts and target.text:
        parts.append(target.text.replace("\n", " "))
    return "\n".join(parts)


def _target_labels(target: Element, doc: LoadedDoc) -> list[str]:
    if target.type == "table":
        return [c for c in doc.tables[target.element_id].columns if len(c) >= 4]
    record = doc.captions.get(target.element_id)
    labels = record.caption.visible_text if record and record.caption else (target.text or "").splitlines()
    return list(dict.fromkeys(x.strip() for x in labels if len(x.strip()) >= 4 and _WORD.search(x.lower())))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def find_links(doc: LoadedDoc, embed: EmbedFn, cfg: Enrich) -> tuple[list[Link], list[str]]:
    """Links for every figure and table, and the ids of those left unlinked."""
    texts = doc.texts()
    order = {e.element_id: i for i, e in enumerate(doc.elements)}
    targets = sorted(doc.figures() + doc.table_elements(), key=lambda e: order[e.element_id])
    links: list[Link] = []
    unlinked: list[str] = []
    pending_related: list[tuple[Element, list[Element]]] = []

    for target in targets:
        label = _target_label(target, doc)
        discussion = [t for t in texts if _norm(t.text) != _norm(label)]

        # 1. Explicit "Figure N" / "Table N".
        number = (_TABLE_NUMBER if target.type == "table" else _FIGURE_NUMBER).match(label or "")
        if number:
            word = r"table" if target.type == "table" else r"(?:figure|fig\.)"
            ref = re.compile(rf"\b{word}\s*{re.escape(number.group(1))}(?![\d.]*\d)", re.IGNORECASE)
            found = [t for t in discussion if ref.search(t.text)]
            if found:
                links += [Link(target_id=target.element_id, text_element_id=t.element_id, method="explicit", score=1.0)
                          for t in found]
                continue

        # 2. A deictic phrase on the same page pointing at this target and no nearer one.
        same_page = [x for x in targets if x.page == target.page]
        found = []
        for t in (t for t in discussion if t.page == target.page):
            if _DEICTIC_BELOW.search(t.text):
                below = [x for x in same_page if x.bbox[1] >= t.bbox[3] - 1]
                if below and min(below, key=lambda x: x.bbox[1]) is target:
                    found.append(t)
            elif _DEICTIC_ABOVE.search(t.text):
                above = [x for x in same_page if x.bbox[3] <= t.bbox[1] + 1]
                if above and max(above, key=lambda x: x.bbox[3]) is target:
                    found.append(t)
        if found:
            links += [Link(target_id=target.element_id, text_element_id=t.element_id, method="deictic", score=1.0)
                      for t in found]
            continue

        # 3. Candidates for the most related nearby paragraph, scored below in one embedding call:
        #    paragraphs within the window on the page, the paragraphs directly before and after,
        #    and paragraphs of the same section on the same or an adjacent page. Headings are skipped.
        pos = order[target.element_id]
        paragraphs = [t for t in discussion if len(t.text.split()) >= MIN_CANDIDATE_WORDS]
        before = [t for t in paragraphs if order[t.element_id] < pos]
        after = [t for t in paragraphs if order[t.element_id] > pos]
        near = [t for t in paragraphs if t.page == target.page and
                min(abs(t.bbox[1] - target.bbox[3]), abs(target.bbox[1] - t.bbox[3])) <= cfg.proximity_window_pt]
        section = [t for t in paragraphs if target.section_path and t.section_path == target.section_path
                   and abs(t.page - target.page) <= 1]
        candidates = list({t.element_id: t for t in near + before[-1:] + after[:1] + section}.values())
        if candidates and _target_text(target, doc):
            pending_related.append((target, candidates))
        else:
            unlinked.append(target.element_id)

    if pending_related:
        wanted = list(dict.fromkeys([_target_text(t, doc) for t, _ in pending_related]
                                    + [c.text for _, cs in pending_related for c in cs]))
        vectors = dict(zip(wanted, embed(wanted)))
        for target, candidates in pending_related:
            tv = vectors[_target_text(target, doc)]
            labels = _target_labels(target, doc)

            def score(t: Element) -> float:
                body = _fold(t.text)
                share = sum(_fold(lbl) in body for lbl in labels) / len(labels) if labels else 0.0
                return _cosine(tv, vectors[t.text]) + cfg.label_weight * share

            best = max(candidates, key=score)
            s = score(best)
            if s >= cfg.link_min_score:
                links.append(Link(target_id=target.element_id, text_element_id=best.element_id, method="related",
                                  score=round(s, 3)))
            else:
                unlinked.append(target.element_id)
    unlinked.sort(key=lambda eid: order[eid])
    return links, unlinked


# ---------------------------------------------------------------- table summaries

_SUMMARY_PROMPT = """Write a two-sentence summary of this table for a search index: say what it compares \
and the main takeaway, using the table's own terms and numbers. Reply with the two sentences only.

Title: {title}
Columns: {columns}
Rows:
{rows}"""


def _table_key(table: Table, model: str) -> str:
    body = json.dumps([table.title, table.columns, table.rows], ensure_ascii=False)
    return hashlib.sha256(f"{model}\n{body}".encode("utf-8")).hexdigest()


def summarize_tables(tables: list[Table], model: str, chat: ChatFn, cache_path: Path) -> dict[str, str]:
    """A summary per table (by element_id), cached by (table content, model) in `cache_path`."""
    cache: dict[str, str] = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                cache[row["key"]] = row["summary"]
    out = {}
    for t in tables:
        key = _table_key(t, model)
        if key not in cache:
            rows = "\n".join(" | ".join(r) for r in t.rows[:25])
            reply = chat(model, _SUMMARY_PROMPT.format(title=t.title or "(none)", columns=" | ".join(t.columns),
                                                       rows=rows))
            cache[key] = " ".join(reply.split())
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "model": model, "summary": cache[key]}, ensure_ascii=False) + "\n")
        out[t.element_id] = cache[key]
    return out
