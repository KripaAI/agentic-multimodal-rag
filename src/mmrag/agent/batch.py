"""`mmrag ask-batch`: a question file answered by one or more models (plan Phase 4, D4).

The file lists questions in order; a follow-up names the earlier question whose thread it
continues (`follows`). Each model gets its own threads. The comparison page shows every
answer page side by side with cost, latency and validator results, for the owner's pick.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, ConfigDict

from mmrag.agent.graph import QueryRun
from mmrag.config import Settings
from mmrag.obs import get_logger

_log = get_logger("mmrag.agent.batch")


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str  # conceptual · visual · quantitative · unanswerable · follow_up
    question: str
    follows: str | None = None  # id of the earlier question whose thread this continues
    note: str | None = None  # what a good answer contains, for the reviewer


@dataclass
class Outcome:
    run: QueryRun | None = None
    page: Path | None = None
    error: str | None = None


def load_questions(path: Path) -> list[Question]:
    qs = [Question.model_validate(q) for q in yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]]
    seen: set[str] = set()
    for q in qs:
        if q.follows and q.follows not in seen:
            raise ValueError(f"{q.id} follows {q.follows}, which is not an earlier question")
        if q.id in seen:
            raise ValueError(f"duplicate question id {q.id}")
        seen.add(q.id)
    return qs


def run_batch(questions: list[Question], models: list[str], settings: Settings,
              ask: Callable[..., QueryRun], page: Callable[[str, QueryRun, Settings], Path],
              max_cost: float | None = None) -> dict[str, dict[str, Outcome]]:
    """Answer every question with every model. One failure is recorded and the batch goes on;
    once the priced spend passes `max_cost` (US$) the remaining questions are skipped."""
    results: dict[str, dict[str, Outcome]] = {}
    spent = 0.0
    for model in models:
        out = results[model] = {}
        for q in questions:
            if max_cost is not None and spent >= max_cost:
                out[q.id] = Outcome(error=f"skipped: cost cap ${max_cost:.2f} reached")
                continue
            prev = out.get(q.follows) if q.follows else None
            thread = prev.run.thread_id if prev and prev.run else None
            try:
                run = ask(q.question, settings, thread_id=thread, model=model)
                spent += run.cost_usd or 0.0
                out[q.id] = Outcome(run=run, page=page(q.question, run, settings))
            except Exception as e:  # noqa: BLE001 - keep the batch going; the error shows on the page
                _log.warning("ask-batch %s %s failed: %s", model, q.id, e)
                out[q.id] = Outcome(error=f"{type(e).__name__}: {e}")
    return results


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def write_comparison_page(questions: list[Question], results: dict[str, dict[str, Outcome]],
                          settings: Settings) -> Path:
    head = "".join(f"<th>{escape(m)}</th>" for m in results)
    summary = []
    for model, out in results.items():
        runs = [o.run for o in out.values() if o.run]
        costs = [r.cost_usd for r in runs if r.cost_usd is not None]
        cost, lat = _median(costs), _median([r.latency_ms / 1000 for r in runs])
        ok = sum(r.validator_result == "ok" for r in runs)
        summary.append(f"<td>median {'$%.4f' % cost if cost is not None else 'cost n/a'} · "
                       f"{'%.1f s' % lat if lat is not None else '-'} · validator ok first try {ok}/{len(out)} · "
                       f"total {'$%.4f' % sum(costs) if costs else 'n/a'}</td>")
    rows = []
    for q in questions:
        cells = []
        for out in results.values():
            o = out.get(q.id)
            if o is None or o.error:
                cells.append(f'<td class="bad">{escape(o.error if o else "not run")}</td>')
                continue
            r = o.run
            cost = f"${r.cost_usd:.4f}" if r.cost_usd is not None else "cost n/a"
            kinds = ", ".join(b.type for b in r.answer.blocks)
            cells.append(f'<td><a href="{escape(o.page.as_uri())}">answer</a> · {kinds}<br><span class="meta">'
                         f"{r.rounds} rounds · {len(r.tool_calls)} tools · {cost} · {r.latency_ms / 1000:.1f} s · "
                         f"{escape(r.validator_result)}</span></td>")
        note = f'<br><span class="meta">{escape(q.note)}</span>' if q.note else ""
        rows.append(f"<tr><th>{escape(q.id)} · {escape(q.kind)}<br>{escape(q.question)}{note}</th>{''.join(cells)}</tr>")
    out_path = settings.resolve(settings.paths.data_dir) / "answers" / f"{datetime.now():%Y%m%d-%H%M%S}-comparison.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Agent model comparison</title><style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1b2330; --muted:#5a6475; --line:#d3d9e1; --bad:#b42318; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#12161c; --panel:#181e26; --ink:#e4e9f0; --muted:#9aa5b4;
  --line:#343d4a; --bad:#f07a6e; }} }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, sans-serif; }}
main {{ max-width:1400px; margin:0 auto; overflow-x:auto; }} table {{ border-collapse:collapse; background:var(--panel); }}
th, td {{ border:1px solid var(--line); padding:6px 10px; text-align:left; vertical-align:top; }}
.meta {{ color:var(--muted); font-size:12px; }} .bad {{ color:var(--bad); }} a {{ color:inherit; }}
</style></head><body><main><h1>Agent model comparison</h1>
<table><tr><th>question</th>{head}</tr><tr><th>summary</th>{''.join(summary)}</tr>{''.join(rows)}</table>
</main></body></html>
""", encoding="utf-8")
    return out_path
