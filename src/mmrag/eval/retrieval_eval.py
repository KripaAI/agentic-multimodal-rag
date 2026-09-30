"""`mmrag eval retrieval`: search-only scores on the golden set, to compare search settings cheaply
(plan Phase 6 tasks 4 and 6) before a full agent evaluation confirms the winner.

Each answerable question is searched with its own wording in the collection its type needs
(mixed: figures and text). A result counts when it covers a reference id (chunk or element).
"""

from __future__ import annotations

import statistics
from typing import Callable

from mmrag.config import Settings
from mmrag.eval.dataset import GoldenItem
from mmrag.retrieval.hybrid import Hit

COLLECTIONS_FOR = {"conceptual": ["text"], "visual": ["figure"], "quantitative": ["table"], "mixed": ["figure", "text"]}
K = 10


def apply_overrides(settings: Settings, overrides: list[str]) -> Settings:
    """`section.field=value` pairs, validated by the config model (e.g. search.keyword_mode=any)."""
    data = settings.model_dump()
    for o in overrides:
        key, _, value = o.partition("=")
        section, _, name = key.partition(".")
        if section not in data or not isinstance(data[section], dict) or name not in data[section]:
            raise ValueError(f"unknown setting {key!r}")
        data[section][name] = {"true": True, "false": False}.get(value.lower(), value)
    return type(settings).model_validate(data)


def retrieval_scores(items: list[GoldenItem], search: Callable[[str, str], list[Hit]],
                     locations: Callable[[list[str]], dict[str, list[dict]]]) -> dict:
    per_q, by_coll = {}, {}
    for item in (i for i in items if i.answerable):
        ref = set(item.reference_ids) | {loc["element_id"] for locs in locations(item.reference_ids).values()
                                         for loc in locs}
        rank = None
        for coll in COLLECTIONS_FOR[item.qtype]:
            for n, h in enumerate(search(item.question, coll)[:K], 1):
                if h.chunk_id in ref or any(loc["element_id"] in ref for loc in h.locations):
                    rank = n if rank is None else min(rank, n)
                    by_coll.setdefault(coll, []).append(n)
                    break
            else:
                by_coll.setdefault(coll, []).append(None)
        per_q[item.question_id] = {"rank": rank, "qtype": item.qtype}

    def stats(ranks: list[int | None]) -> dict:
        return {"hit@5": round(sum(r is not None and r <= 5 for r in ranks) / len(ranks), 4),
                "hit@10": round(sum(r is not None for r in ranks) / len(ranks), 4),
                "mrr@10": round(statistics.fmean((1 / r) if r else 0.0 for r in ranks), 4),
                "questions": len(ranks)}
    return {"overall": stats([q["rank"] for q in per_q.values()]),
            "by_collection": {c: stats(r) for c, r in sorted(by_coll.items())}, "per_question": per_q}
