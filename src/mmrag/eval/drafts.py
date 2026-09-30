"""`mmrag eval draft-questions`: candidate golden questions drafted from the corpus by the judge
model, for the owner to review (spec §9: a question joins the golden set only after review).

Evidence is sampled evenly across the books; each draft is written from one piece of evidence
only, so its reference answer and reference ids are known. Chart drafts keep only values printed
in the source table.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from html import escape
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mmrag.agent.ledger import numbers_in
from mmrag.eval.dataset import GoldenItem

# spec §9 mix, with spares for the owner to choose from
QUOTA = {"conceptual": 22, "visual": 15, "quantitative": 12, "mixed": 6}
TOOLS = {"conceptual": ["search_text"], "visual": ["search_figures"], "quantitative": ["make_chart"],
         "mixed": ["search_text", "search_figures"]}
SOURCE = {"conceptual": "text", "visual": "figure", "quantitative": "table", "mixed": "figure"}

# Hand-written; each was checked against the corpus text before use (see the draft note).
UNANSWERABLE = [
    ("What was the total training cost of Qwen3-14B in US dollars?", "Qwen3-14B"),
    ("How many parameters does GPT-5 have?", "GPT-5"),
    ("What is the price per million output tokens of Claude Opus?", "Claude"),
    ("Which paper won the NeurIPS 2024 best paper award?", "NeurIPS"),
    ("How many tonnes of CO2 were emitted to train GPT-3?", "CO2"),
]

INSTRUCTIONS = {
    "conceptual": "Write one question a learner would ask whose full answer is in the EVIDENCE, and a 2-4 sentence "
                  "reference answer using only the EVIDENCE.",
    "visual": "The EVIDENCE describes one figure. Write one question asking to see or explain what this figure shows "
              "(e.g. 'Show me ...' or 'What does the diagram of ... show?'), and a 2-3 sentence reference answer "
              "using only the EVIDENCE.",
    "quantitative": "The EVIDENCE is a table. Write one question asking to compare or chart 2-8 numbers with the same "
                    "unit from it, a reference answer quoting those numbers, and expected_chart_values: exactly those "
                    "numbers as printed (no unit conversion, percentages as printed).",
    "mixed": "The EVIDENCE describes a figure and the paragraph that discusses it. Write one question that needs both "
             "an explanation and the figure, and a 3-4 sentence reference answer using only the EVIDENCE.",
}
RULES = ("Do not mention page, figure or table numbers, file names, or 'the document'. Ask as someone who has not "
         "seen the source: never write 'the table', 'the slide', 'shown above', 'the outer ring' or similar (a "
         "visual question may ask 'Show me a diagram of ...'). Do not copy a sentence verbatim into the question. "
         "Leave expected_chart_values empty unless asked for it.")


class DraftOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    reference_answer: str
    expected_chart_values: list[float]


@dataclass
class Evidence:
    chunk_id: str
    collection: str
    source_file: str
    element_ids: list[str]
    text: str


def pick_evidence(pool: list[Evidence], n: int, seed: int) -> list[Evidence]:
    """`n` pieces, round-robin across books (shuffled within each book)."""
    rng = random.Random(seed)
    by_book: dict[str, list[Evidence]] = {}
    for e in pool:
        by_book.setdefault(e.source_file, []).append(e)
    for items in by_book.values():
        rng.shuffle(items)
    books = sorted(by_book)
    rng.shuffle(books)
    out: list[Evidence] = []
    while len(out) < n and any(by_book.values()):
        for b in books:
            if by_book[b] and len(out) < n:
                out.append(by_book[b].pop())
    return out


def draft_one(qid: str, qtype: str, ev: Evidence, judge) -> GoldenItem | None:
    msg = f"{INSTRUCTIONS[qtype]} {RULES}\n\nEVIDENCE:\n{ev.text[:3000]}"
    out = judge.structured([{"role": "system", "content": "You write evaluation questions for a document QA system."},
                            {"role": "user", "content": msg}], DraftOut)
    values = out.expected_chart_values if qtype == "quantitative" else []
    if qtype == "quantitative":
        printed = numbers_in(ev.text)
        if not values or any(not any(abs(v - p) < 1e-9 for p in printed) for v in values):
            return None  # a value not printed in the table: the chart could never be verified
    ref = ev.element_ids if ev.collection != "text" else [ev.chunk_id]
    return GoldenItem(question_id=qid, question=out.question, qtype=qtype, reference_answer=out.reference_answer,
                      reference_ids=ref, expected_figure_ids=ev.element_ids[:1] if qtype in ("visual", "mixed") else [],
                      expected_chart_values=values, expected_tool_calls=TOOLS[qtype],
                      source_file=ev.source_file, note=f"draft from {ev.source_file} ({ev.chunk_id})")


def load_pool(settings) -> dict[str, list[Evidence]]:
    """Candidate evidence per collection: substantial passages, figures (not decorative), tables with numbers."""
    from mmrag import db

    with db.connect(settings) as conn:
        rows = conn.execute(
            "SELECT sc.chunk_id, sc.collection, d.source_file, sc.element_ids, sc.dense_text FROM search_chunks sc "
            "JOIN documents d USING (doc_id)").fetchall()
    pool: dict[str, list[Evidence]] = {"text": [], "figure": [], "table": []}
    for cid, coll, src, eids, text in rows:
        e = Evidence(cid, coll, src, list(eids), text)
        if coll == "text" and len(text.split()) >= 80:
            pool["text"].append(e)
        elif coll == "figure" and "decorative" not in text.lower()[:200] and len(text.split()) >= 30:
            pool["figure"].append(e)
        elif coll == "table" and len(numbers_in(text)) >= 4:
            pool["table"].append(e)
    return pool


def unanswerable_items(settings) -> list[GoldenItem]:
    """The hand-written unanswerable questions whose key term is absent from the corpus text."""
    from mmrag import db

    items = []
    with db.connect(settings) as conn:
        for n, (q, term) in enumerate(UNANSWERABLE, 1):
            hits = conn.execute("SELECT count(*) FROM search_chunks WHERE dense_text ILIKE %s", (f"%{term}%",)).fetchone()[0]
            note = f"hand-written; '{term}' appears in {hits} passages" + (" - check it is really not answered" if hits else "")
            items.append(GoldenItem(question_id=f"u{n}", question=q, qtype="unanswerable",
                                    reference_answer="The documents do not contain this information.",
                                    answerable=False, note=note))
    return items


def draft_questions(settings, judge, seed: int = 7, progress=print) -> list[tuple[GoldenItem, Evidence | None]]:
    pool = load_pool(settings)
    drafts: list[tuple[GoldenItem, Evidence | None]] = []
    prefix = {"conceptual": "c", "visual": "v", "quantitative": "n", "mixed": "m"}
    used: set[str] = set()  # each piece of evidence backs one question only
    for qtype, n in QUOTA.items():
        candidates = [e for e in pool[SOURCE[qtype]] if e.chunk_id not in used]
        made = 0
        for ev in pick_evidence(candidates, len(candidates), seed):  # keep trying until the quota is met
            if made == n:
                break
            used.add(ev.chunk_id)
            item = draft_one(f"{prefix[qtype]}{made + 1:02d}", qtype, ev, judge)
            if item:
                drafts.append((item, ev))
                made += 1
        progress(f"{qtype}: {made} drafts")
    drafts += [(u, None) for u in unanswerable_items(settings)]
    return drafts


def write_drafts(drafts, jsonl: Path, html_out: Path, data_dir: Path, asset_of: dict[str, str]) -> None:
    jsonl.write_text("\n".join(i.model_dump_json(exclude_defaults=False) for i, _ in drafts) + "\n", encoding="utf-8")
    rows = []
    for item, ev in drafts:
        img = asset_of.get(item.expected_figure_ids[0]) if item.expected_figure_ids else None
        img_html = f'<img src="{(data_dir / img).as_uri()}" alt="">' if img else ""
        extra = f"<p class=\"meta\">chart values: {item.expected_chart_values}</p>" if item.expected_chart_values else ""
        rows.append(f"<section><h3>{escape(item.question_id)} · {escape(item.qtype)}</h3>"
                    f"<p class=\"q\">{escape(item.question)}</p><p><b>Reference:</b> {escape(item.reference_answer)}</p>"
                    f"{extra}<p class=\"meta\">{escape(item.note or '')}</p>{img_html}"
                    + (f"<details><summary>evidence</summary><pre>{escape(ev.text[:2000])}</pre></details>" if ev else "")
                    + "</section>")
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Golden set drafts</title><style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1b2330; --muted:#5a6475; --line:#d3d9e1; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#12161c; --panel:#181e26; --ink:#e4e9f0; --muted:#9aa5b4; --line:#343d4a; }} }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, sans-serif; }}
main {{ max-width:900px; margin:0 auto; display:grid; gap:12px; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 14px; }}
h3 {{ margin:0; font-size:13px; color:var(--muted); }} .q {{ font-size:16px; margin:4px 0; }} .meta {{ color:var(--muted); font-size:12px; }}
img {{ max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; }} pre {{ white-space:pre-wrap; font-size:12px; }}
</style></head><body><main><h1>Golden set drafts ({len(drafts)})</h1>
<p class="meta">Pick 40: 15 conceptual, 10 visual, 8 quantitative, 4 mixed, 3 unanswerable. Edit any wording or reference answer.</p>
{''.join(rows)}</main></body></html>
""", encoding="utf-8")


def coverage(golden_file: Path) -> dict[str, dict[str, int]]:
    """Questions per book and type (unanswerable ones have no book)."""
    from mmrag.eval.dataset import load_golden_set

    out: dict[str, dict[str, int]] = {}
    for item in load_golden_set(golden_file)[0]:
        book = out.setdefault(item.source_file or "(none)", {})
        book[item.qtype] = book.get(item.qtype, 0) + 1
    return out


def promote(drafts_file: Path, golden_file: Path, ids: list[str]) -> int:
    """Copy the chosen drafts (in the given order) into the golden set, validated on the way."""
    from mmrag.eval.dataset import load_golden_set

    by_id = {json.loads(line)["question_id"]: line for line in drafts_file.read_text(encoding="utf-8").splitlines() if line}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"not in the drafts: {missing}")
    golden_file.write_text("\n".join(by_id[i] for i in ids) + "\n", encoding="utf-8")
    return len(load_golden_set(golden_file)[0])
