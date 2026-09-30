"""`mmrag eval run` (LLD §5.10): answer every golden question with the production agent, score it,
store the scores, and apply the regression gate against the latest baseline."""

from __future__ import annotations

import hashlib
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from psycopg.types.json import Jsonb

from mmrag.agent.graph import QueryRun
from mmrag.config import PROJECT_ROOT, Settings
from mmrag.eval import custom_metrics as cm
from mmrag.eval.dataset import GoldenItem
from mmrag.eval.judge import JUDGE_PROMPT_VERSION
from mmrag.obs import get_logger, get_tracer

_log = get_logger("mmrag.eval")
Score = tuple[float | None, dict]
JUDGED = ["faithfulness", "multimodal_faithfulness", "response_relevancy", "context_precision", "context_recall",
          "factual_correctness"]
DETERMINISTIC = ["tool_call_accuracy", "figure_hit", "citation_accuracy", "chart_numeric", "refusal"]
MUST_BE_PERFECT = ["chart_numeric", "refusal"]  # spec §9: 100% metrics
GATED = [*JUDGED, "tool_call_accuracy", "figure_hit", "citation_accuracy"]
MAX_CONTEXTS = 20
TRANSIENT = ("APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError")


def _ask_with_retry(ask, question: str, settings, thread_id, model, attempts: int = 3) -> QueryRun:
    """A network or rate-limit error is retried (after 10 s, then 30 s); anything else is a real failure."""
    for n in range(attempts):
        try:
            return ask(question, settings, thread_id=thread_id, model=model)
        except Exception as e:  # noqa: BLE001 - re-raised unless transient
            if type(e).__name__ not in TRANSIENT or n == attempts - 1:
                raise
            _log.warning("transient %s, retrying: %s", type(e).__name__, e)
            time.sleep(10 if n == 0 else 30)
    raise AssertionError("unreachable")


def _answer_text(run: QueryRun) -> str:
    return "\n\n".join(b.content.get("markdown", "") for b in run.answer.blocks if b.type == "text")


def score_item(item: GoldenItem, run: QueryRun, metrics, store, data_dir: Path) -> dict[str, Score]:
    """Every metric that applies to this question (spec §9)."""
    scores: dict[str, Score] = {}
    text = _answer_text(run)
    blocks = run.answer.blocks
    if not item.answerable:
        scores["refusal"] = (cm.refusal(False, run.not_found, text), {"not_found_flag": run.not_found})
    else:
        rankings = [t["result_ids"] for t in run.tool_calls if t.get("result_ids")]
        texts = {e["id"]: e["text"] for e in run.evidence if not e["id"].startswith("compute:")}
        ranked = [i for r in rankings for i in r if i in texts]  # searched evidence first, then the rest
        order = list(dict.fromkeys(ranked + list(texts)))[:MAX_CONTEXTS]
        contexts = [(i, texts[i]) for i in order]
        images = [str(data_dir / b.content["asset_path"]) for b in blocks
                  if b.type == "image" and b.content.get("asset_path")]
        scores["faithfulness"] = metrics.faithfulness(item.question, text, contexts)
        if images:
            scores["multimodal_faithfulness"] = metrics.faithfulness(item.question, text, contexts, images=images)
        scores["response_relevancy"] = metrics.response_relevancy(item.question, text)
        scores["context_precision"] = metrics.context_precision(item.question, item.reference_answer, contexts,
                                                                rankings=rankings or None)
        scores["context_recall"] = metrics.context_recall(item.question, item.reference_answer, contexts)
        scores["factual_correctness"] = metrics.factual_correctness(text, item.reference_answer)

        tools = [t["name"] for t in run.tool_calls]
        scores["tool_call_accuracy"] = (cm.tool_call_accuracy(tools, item.expected_tool_calls), {"called": tools})
        shown = [b.content.get("element_id") for b in blocks if b.type == "image"]
        scores["figure_hit"] = (cm.figure_hit(shown, item.expected_figure_ids), {"shown": shown})
        charts = [cm.chart_values(b.content.get("data_table") or []) for b in blocks if b.type == "chart"]
        scores["chart_numeric"] = cm.chart_numeric(charts, item.expected_chart_values, percent_expected=True)
        # a citation is right when it is a reference passage or on the same page as one (the drafted
        # reference is a single passage; its neighbours on the page support the same point)
        ref_locs = [loc for locs in store.locations(item.reference_ids).values() for loc in locs]
        ref = set(item.reference_ids) | {loc["element_id"] for loc in ref_locs} \
            | {f"{loc['source_file']}#p{loc['page']}" for loc in ref_locs if "page" in loc}
        cited = [{c.id} | {loc.element_id for loc in c.locations} | {f"{loc.source_file}#p{loc.page}" for loc in c.locations}
                 for b in blocks for c in b.citations]
        scores["citation_accuracy"] = (cm.citation_accuracy(cited, ref), {"cited": sorted(c.id for b in blocks
                                                                                          for c in b.citations)})
    scores["latency_s"] = (run.latency_ms / 1000, {})
    scores["cost_usd"] = (run.cost_usd, {})
    return {k: v for k, v in scores.items() if v[0] is not None or k == "cost_usd"}


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(results: dict[str, dict[str, Score]], items: list[GoldenItem]) -> dict:
    qtype = {i.question_id: i.qtype for i in items}
    names = sorted({m for r in results.values() for m in r} - {"latency_s", "cost_usd"})
    pick = lambda m, ids: [results[q][m][0] for q in ids if m in results[q] and results[q][m][0] is not None]  # noqa: E731
    by_type = {t: {m: _mean(pick(m, [q for q in results if qtype[q] == t])) for m in names}
               for t in sorted(set(qtype[q] for q in results))}
    latency = pick("latency_s", results)
    costs = pick("cost_usd", results)
    return {"metrics": {m: _mean(pick(m, results)) for m in names},
            "by_type": {t: {m: v for m, v in d.items() if v is not None} for t, d in by_type.items()},
            "questions": len(results),
            "median_latency_s": statistics.median(latency) if latency else None,
            "p90_latency_s": statistics.quantiles(latency, n=10)[-1] if len(latency) >= 2 else (latency or [None])[0],
            "median_cost_usd": statistics.median(costs) if costs else None,
            "total_cost_usd": round(sum(costs), 6) if costs else None}


def gate(summary: dict, baseline: dict | None, tolerance: float) -> tuple[bool, list[str]]:
    """Fails on any 100% metric below 1.0, or a gated metric more than `tolerance` below the baseline."""
    failures = []
    now = summary["metrics"]
    for m in MUST_BE_PERFECT:
        if now.get(m) is not None and now[m] < 1.0:
            failures.append(f"{m} = {now[m]:.3f} (must be 1.0)")
    if baseline:
        for m in GATED:
            before, after = baseline["metrics"].get(m), now.get(m)
            if before is not None and after is not None and after < before - tolerance:
                failures.append(f"{m}: {after:.3f} vs baseline {before:.3f} (tolerance {tolerance})")
    return not failures, failures


@dataclass
class RunOutcome:
    run_id: str
    summary: dict
    passed: bool
    failures: list[str] = field(default_factory=list)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except OSError:
        return None


def run_eval(settings: Settings, items: list[GoldenItem], golden_version: str, ask: Callable[..., QueryRun],
             metrics, store, agent_model: str, baseline: bool = False, label: str | None = None,
             progress: Callable[[str], None] = print) -> RunOutcome:
    """Answer and score every item (follow-ups reuse their parent's thread), store everything."""
    from mmrag import db

    config_hash = hashlib.sha256((PROJECT_ROOT / "config.yaml").read_bytes()).hexdigest()[:12] \
        if (PROJECT_ROOT / "config.yaml").is_file() else "n/a"
    data_dir = settings.resolve(settings.paths.data_dir)
    with get_tracer("mmrag.eval").start_as_current_span("eval.run") as span, db.connect(settings) as conn:
        trace_id = format(span.get_span_context().trace_id, "032x")
        run_id = str(conn.execute(
            "INSERT INTO eval_runs (git_commit, config_hash, agent_model, judge_model, ragas_version, "
            "golden_set_version, is_baseline, label, trace_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING run_id",
            (_git_commit(), config_hash, agent_model, settings.eval.judge_model, JUDGE_PROMPT_VERSION, golden_version,
             baseline, label, trace_id)).fetchone()[0])
        conn.commit()
        threads: dict[str, str] = {}
        results: dict[str, dict[str, Score]] = {}
        for n, item in enumerate(items, 1):
            try:
                run = _ask_with_retry(ask, item.question, settings,
                                      threads.get(item.follows) if item.follows else None, agent_model)
                threads[item.question_id] = run.thread_id
                scores, qtrace = score_item(item, run, metrics, store, data_dir), run.trace_id
            except Exception as e:  # noqa: BLE001 - one failure must not end the run; it scores 0
                _log.warning("eval %s failed: %s", item.question_id, e)
                scores, qtrace = {"run_error": (0.0, {"error": f"{type(e).__name__}: {e}"})}, None
            results[item.question_id] = scores
            conn.cursor().executemany(
                "INSERT INTO eval_results (run_id, question_id, metric, score, details, trace_id) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                [(run_id, item.question_id, m, s, Jsonb(d), qtrace) for m, (s, d) in scores.items()])
            conn.commit()
            shown = ", ".join(f"{m} {s:.2f}" for m, (s, _) in scores.items() if s is not None and m in (*JUDGED, *DETERMINISTIC))
            progress(f"[{n}/{len(items)}] {item.question_id}: {shown}")

        summary = summarize(results, items)
        judge_usage = getattr(getattr(metrics, "judge", None), "usage", None)
        if judge_usage:
            summary["judge_tokens"] = {"input": judge_usage[0], "output": judge_usage[1]}
        prev = conn.execute("SELECT summary FROM eval_runs WHERE is_baseline AND run_id <> %s AND summary IS NOT NULL "
                            "ORDER BY started_at DESC LIMIT 1", (run_id,)).fetchone()
        passed, failures = gate(summary, prev[0] if prev else None, settings.eval.regression_tolerance)
        summary["gate_failures"] = failures
        conn.execute("UPDATE eval_runs SET finished_at = now(), summary = %s, passed = %s WHERE run_id = %s",
                     (Jsonb(summary), passed, run_id))
        conn.commit()
        span.set_attribute("mmrag.eval.passed", passed)
    return RunOutcome(run_id=run_id, summary=summary, passed=passed, failures=failures)
