"""Evaluation report (LLD §5.10 step 8): averages vs targets, per question type, the worst questions
with the judge's reasons, and every question with links to its trace."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from mmrag.config import Settings

# spec §9 targets (v1, provisional)
TARGETS = {"faithfulness": 0.90, "multimodal_faithfulness": 0.85, "response_relevancy": 0.85,
           "context_precision": 0.80, "context_recall": 0.80, "factual_correctness": 0.75,
           "tool_call_accuracy": 0.80, "chart_numeric": 1.0, "figure_hit": 0.80, "citation_accuracy": 0.90,
           "refusal": 1.0}
WORST_BY = ["factual_correctness", "faithfulness", "context_recall"]


def _fmt(v) -> str:
    return "–" if v is None else f"{float(v):.2f}"


def _reasons(details: dict) -> str:
    """The judge's failed claims or verdicts, for the worst-question view and the spot-check."""
    out = []
    for key in ("claims", "answer_claims", "reference_claims"):
        out += [f"✗ {c['statement']} — {c['reason']}" for c in details.get(key, []) if not c.get("supported")]
    out += [f"✗ {v['reason']}" for v in details.get("verdicts", []) if not v.get("useful") and v.get("reason")][:3]
    for key in ("wrong", "missing", "error"):
        if details.get(key):
            out.append(f"{key}: {details[key]}")
    return "<br>".join(escape(x) for x in out[:8])


def write_report(settings: Settings, run_id: str) -> Path:
    from mmrag import db

    with db.connect(settings) as conn:
        run = conn.execute("SELECT started_at, git_commit, agent_model, judge_model, golden_set_version, is_baseline, "
                           "label, summary, passed, trace_id FROM eval_runs WHERE run_id = %s", (run_id,)).fetchone()
        if run is None:
            raise ValueError(f"no evaluation run {run_id}")
        rows = conn.execute("SELECT question_id, metric, score, details, trace_id FROM eval_results WHERE run_id = %s "
                            "ORDER BY question_id, metric", (run_id,)).fetchall()
    started, commit, agent_model, judge, version, is_baseline, label, summary, passed, trace = run
    summary = summary or {}
    endpoint = settings.observability.otlp_traces_endpoint or ""
    phoenix = endpoint.rsplit("/v1/", 1)[0] if endpoint else ""
    link = lambda t: f'<a href="{escape(phoenix)}/redirects/traces/{t}">trace</a>' if phoenix and t else ""  # noqa: E731

    per_q: dict[str, dict] = {}
    for qid, metric, score, details, qtrace in rows:
        per_q.setdefault(qid, {"trace": qtrace, "scores": {}, "details": {}})
        per_q[qid]["scores"][metric] = None if score is None else float(score)
        per_q[qid]["details"][metric] = details or {}

    metric_rows = "".join(
        f"<tr><td>{escape(m)}</td><td>{_fmt(v)}</td><td>{_fmt(TARGETS.get(m))}</td>"
        f"<td class=\"{'ok' if m in TARGETS and v is not None and v >= TARGETS[m] else 'bad' if m in TARGETS else ''}\">"
        f"{'meets' if m in TARGETS and v is not None and v >= TARGETS[m] else 'below' if m in TARGETS else ''}</td></tr>"
        for m, v in summary.get("metrics", {}).items())
    types = summary.get("by_type", {})
    cols = sorted({m for d in types.values() for m in d})
    type_rows = "".join(f"<tr><td>{escape(t)}</td>{''.join(f'<td>{_fmt(d.get(m))}</td>' for m in cols)}</tr>"
                        for t, d in types.items())

    def badness(q):
        s = q["scores"]
        vals = [s[m] for m in WORST_BY if s.get(m) is not None]
        return (min(vals) if vals else 1.0) + (-1 if "run_error" in s else 0)
    worst = sorted(per_q.items(), key=lambda kv: badness(kv[1]))[:5]
    worst_html = "".join(
        f"<section><h3>{escape(q)} · {link(d['trace'])}</h3><p>"
        + " · ".join(f"{escape(m)} {_fmt(v)}" for m, v in d["scores"].items() if m not in ("latency_s", "cost_usd"))
        + "</p><p class=\"why\">" + "<br>".join(_reasons(d["details"][m]) for m in d["details"] if _reasons(d["details"][m]))
        + "</p></section>" for q, d in worst)
    all_metrics = sorted({m for d in per_q.values() for m in d["scores"]})
    cells = lambda s: "".join(f"<td>{_fmt(s.get(m))}</td>" for m in all_metrics)  # noqa: E731
    q_rows = "".join(f"<tr><td>{escape(q)}</td>{cells(d['scores'])}<td>{link(d['trace'])}</td></tr>"
                     for q, d in per_q.items())
    gate = "PASSED" if passed else "FAILED"
    failures = "".join(f"<li>{escape(f)}</li>" for f in summary.get("gate_failures", []))
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Evaluation report</title><style>
:root {{ --bg:#f6f7f9; --panel:#fff; --ink:#1b2330; --muted:#5a6475; --line:#d3d9e1; --ok:#0f766e; --bad:#b42318; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#12161c; --panel:#181e26; --ink:#e4e9f0; --muted:#9aa5b4;
  --line:#343d4a; --ok:#3cc9b8; --bad:#f07a6e; }} }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, sans-serif; }}
main {{ max-width:1300px; margin:0 auto; display:grid; gap:14px; }} .wrap {{ overflow-x:auto; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 14px; }}
table {{ border-collapse:collapse; background:var(--panel); }} th, td {{ border:1px solid var(--line); padding:4px 8px; }}
.ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }} .meta, .why {{ color:var(--muted); font-size:13px; }}
h1 {{ font-size:22px; margin:0; }} h2 {{ font-size:16px; margin:8px 0 4px; }} h3 {{ font-size:14px; margin:0; }}
</style></head><body><main>
<h1>Evaluation {'baseline ' if is_baseline else ''}run · <span class="{'ok' if passed else 'bad'}">{gate}</span></h1>
<p class="meta">{escape(str(started))} · commit {escape(commit or '?')} · agent {escape(agent_model)} · judge {escape(judge)}
· golden set {escape(version)} · {summary.get('questions', 0)} questions{(' · ' + escape(label)) if label else ''}
· median {_fmt(summary.get('median_latency_s'))} s (p90 {_fmt(summary.get('p90_latency_s'))} s)
· median cost ${_fmt(summary.get('median_cost_usd'))} · total ${_fmt(summary.get('total_cost_usd'))} · {link(trace)}</p>
{f'<section><h2>Gate failures</h2><ul>{failures}</ul></section>' if failures else ''}
<h2>Metrics vs targets (spec §9)</h2><div class="wrap"><table><tr><th>metric</th><th>score</th><th>target</th><th></th></tr>{metric_rows}</table></div>
<h2>By question type</h2><div class="wrap"><table><tr><th>type</th>{''.join(f'<th>{escape(m)}</th>' for m in cols)}</tr>{type_rows}</table></div>
<h2>Five weakest questions, with the judge's reasons</h2>{worst_html}
<h2>All questions</h2><div class="wrap"><table><tr><th>question</th>{''.join(f'<th>{escape(m)}</th>' for m in all_metrics)}<th></th></tr>{q_rows}</table></div>
<details><summary class="meta">raw summary</summary><pre>{escape(json.dumps(summary, indent=1))}</pre></details>
</main></body></html>
"""
    out = settings.resolve(settings.paths.data_dir) / "eval" / f"eval_{str(started)[:19].replace(':', '').replace(' ', '_')}_{run_id[:8]}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
