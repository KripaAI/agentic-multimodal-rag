"""Automatic benchmark for the reading writer: 100 held-out birth charts, a full reading and two questions each.

The writer's first draft (before the app's repair turn) is checked by the same code checks the app uses, so the
benchmark needs no judge and costs nothing per run beyond the GPU time. It measures the model, not the app's
repair and removal steps, and the drafts with flagged sentences become DPO candidates: rejected = the draft,
chosen = the draft with those sentences removed by code (no other model writes anything).

    python scripts/chart_benchmark.py build   --out data/eval/astro_bench/v1          # charts + exact app prompts
    python scripts/chart_benchmark.py score   --bench data/eval/astro_bench/v1 --outputs <eval_outputs.jsonl> --name v4
    python scripts/chart_benchmark.py compare --bench data/eval/astro_bench/v1 v4 <other>

`build` needs the library database (MMRAG_CONFIG=config.astro.yaml) because each prompt carries the passages the app
would retrieve. The prompts run on Kaggle with jyotish-lora's gguf_eval notebook (EXTRA_FILE = the prompts file).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ON = date(2026, 10, 13)          # every reading is written "as of" this date, so runs are comparable
SEED = 2026_10_13
CITIES = [("New Delhi", 28.61, 77.21), ("Mumbai", 19.08, 72.88), ("Kolkata", 22.57, 88.36), ("Chennai", 13.08, 80.27),
          ("Bengaluru", 12.97, 77.59), ("Hyderabad", 17.39, 78.49), ("Patna", 25.59, 85.14), ("Jaipur", 26.91, 75.79),
          ("Lucknow", 26.85, 80.95), ("Ahmedabad", 23.02, 72.57), ("Pune", 18.52, 73.86), ("Varanasi", 25.32, 82.97),
          ("Bhopal", 23.26, 77.41), ("Guwahati", 26.14, 91.74), ("Thiruvananthapuram", 8.52, 76.94),
          ("Vaishali", 25.68, 85.20), ("Amritsar", 31.63, 74.87), ("Indore", 22.72, 75.86), ("Nagpur", 21.15, 79.09),
          ("Bhubaneswar", 20.30, 85.82)]
# age bands (years on ON) and how many charts each gets: children and elders are where the rules differ most
BANDS = [("child", 1, 13, 15), ("young person", 14, 24, 15), ("adult", 25, 59, 55), ("elder", 60, 85, 15)]
QUESTIONS = {
    "child": ["How will the child's career be?", "What is the child's temperament and how can we support learning?",
              "What will the coming years bring for the child?"],
    "young person": ["What career should I choose?", "When will I get married?", "What will the coming years bring?",
                     "Which gemstone or remedy should I consider?"],
    "adult": ["What career should I choose?", "How will my married life be?", "What will the coming years bring?",
              "Which gemstone or remedy should I consider?", "How is my financial future?"],
    "elder": ["What will the coming years bring?", "How is my health in general?", "How will my family life be?"],
}


def _training_births() -> set[str]:
    """'Born YYYY-MM-DD HH:MM' of every chart in the v4 training data, to keep the benchmark held out."""
    root = Path("C:/Users/krish/Projects/jyotish-lora/data")
    found: set[str] = set()
    for f in [*root.glob("sft_v4/*.jsonl"), *root.glob("sft_v2/*.jsonl")]:
        found |= set(re.findall(r"Born \d{4}-\d\d-\d\d \d\d:\d\d", f.read_text(encoding="utf-8", errors="ignore")))
    return found


def build(out: Path, n_per_band_scale: float = 1.0) -> None:
    from mmrag.astro.analysis import life_stage, neecha_bhanga
    from mmrag.astro.natal import compute_chart, parse_birth
    from mmrag.astro.reading import Written, write_reading
    from mmrag.config import get_settings

    settings = get_settings()
    rng = random.Random(SEED)
    seen = _training_births()
    print(f"{len(seen)} births in the training data are excluded")
    charts, prompts = [], []
    for stage_name, lo, hi, n in BANDS:
        made = 0
        while made < round(n * n_per_band_scale):
            age = rng.randint(lo, hi)
            born = date(ON.year - age - 1, rng.randint(1, 12), rng.randint(1, 28))
            if life_stage(born, ON).stage != stage_name:
                continue
            hh, mm = rng.randint(0, 23), rng.randint(0, 59)
            if f"Born {born.isoformat()} {hh:02d}:{mm:02d}" in seen:
                continue
            city, lat, lon = rng.choice(CITIES)
            chart = compute_chart(parse_birth(born.isoformat(), f"{hh:02d}:{mm:02d}", "Asia/Kolkata", lat, lon, None, city))
            cid = f"c{len(charts):03d}"
            p = chart.placements
            flags = sorted({"retrograde" for b in p.values() if getattr(b, "retrograde", False)}
                           | {"debilitated" for b in p.values() if b.dignity == "debilitated"}
                           | ({"cancelled debility"} if any(nb.classical for nb in neecha_bhanga(chart)) else set())
                           | ({"lagna near a cusp"} if chart.lagna.degree < 2 or chart.lagna.degree > 28 else set()))
            questions = [None] + rng.sample(QUESTIONS[stage_name], 2)
            charts.append({"id": cid, "date": born.isoformat(), "time": f"{hh:02d}:{mm:02d}", "place": city,
                           "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata", "stage": stage_name,
                           "age": life_stage(born, ON).age, "flags": flags, "questions": questions})
            for k, q in enumerate(questions):
                captured = {}

                def capture(messages, captured=captured):
                    captured["messages"] = messages
                    return Written(text="", model="capture")

                write_reading(settings, chart, question=q, write=capture, on=ON)
                prompts.append({"set": "bench", "id": f"{cid}-{k}", "task": "reading" if q is None else "question",
                                "route": q or "full reading", "messages": captured["messages"],
                                "max_new": settings.writer.max_tokens})
            made += 1
            print(f"{cid} {stage_name:12} {born} {hh:02d}:{mm:02d} {city:18} flags={flags}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "charts.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in charts) + "\n", encoding="utf-8")
    (out / "bench_prompts.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in prompts) + "\n",
                                             encoding="utf-8")
    longest = max(sum(len(m["content"]) for m in p["messages"]) for p in prompts)
    print(f"{len(charts)} charts, {len(prompts)} prompts -> {out}; longest prompt {longest} chars "
          f"(~{longest // 3.3:.0f} tokens); flags: {dict(Counter(f for c in charts for f in c['flags']))}")


# ---------------------------------------------------------------- scoring

CATEGORIES = [("lordship", r"lord of|th lord|rules"), ("aspect", r"aspect"), ("dignity", r"exalted|debilitated|own sign"),
              ("cancellation", r"debility is cancelled|cancell"), ("dasha", r"mahadasha|sub-period|dasha|antardasha"),
              ("yoga", r"yoga"), ("house or sign", r"house|Lagna| in [A-Z][a-z]+;"), ("counting", r"\bfrom\b"),
              ("vargottama", r"Vargottama"), ("age milestone", r"school"), ("career: no 10th house", r"10th house")]


def category(problem: str) -> str:
    text = problem.split(": the text ", 1)[-1]
    return next((name for name, pat in CATEGORIES if re.search(pat, text)), "other")


def score(bench: Path, outputs: Path, name: str) -> None:
    from mmrag.astro.analysis import life_stage
    from mmrag.astro.compose import CAREER_QUESTION, collapse_repetition, strip_factual_sections
    from mmrag.astro.natal import compute_chart, parse_birth
    from mmrag.astro.reading import check_analysis, check_positions, remove_flagged_sentences

    charts = {c["id"]: c for c in map(json.loads, (bench / "charts.jsonl").open(encoding="utf-8"))}
    prompts = {p["id"]: p for p in map(json.loads, (bench / "bench_prompts.jsonl").open(encoding="utf-8"))}
    answers = {r["id"]: r for r in map(json.loads, outputs.open(encoding="utf-8")) if r.get("set") == "bench"}
    rows, dpo = [], []
    cache = {}
    for pid, prompt in prompts.items():
        if pid not in answers:
            continue
        c = charts[pid.rsplit("-", 1)[0]]
        if c["id"] not in cache:
            cache[c["id"]] = compute_chart(parse_birth(c["date"], c["time"], c["timezone"], c["latitude"],
                                                       c["longitude"], None, c["place"]))
        chart, stage = cache[c["id"]], life_stage(date.fromisoformat(c["date"]), ON)
        raw = answers[pid]["answer"]
        body, looped = collapse_repetition(strip_factual_sections(raw))
        problems = list(dict.fromkeys(check_positions(body, chart) + check_analysis(body, chart, stage)))
        question = prompt["route"] if prompt["task"] == "question" else None
        career = bool(question and CAREER_QUESTION.search(question))
        if career:
            problems = [p for p in problems if "discusses career in a reading" not in p]
            if body and not re.search(r"\b10th\b|\btenth\b", body, re.I):
                problems.append("Analysis check: the text career: no 10th house discussed.")
        sentences = max(1, len(re.findall(r"[.!?](?:\s|$)", body)))
        rows.append({"id": pid, "chart": c["id"], "stage": c["stage"], "task": prompt["task"], "route": prompt["route"],
                     "flags": c["flags"], "problems": problems, "n_problems": len(problems), "sentences": sentences,
                     "words": len(body.split()), "citations": len(re.findall(r"\[\s*P\s*\d", body)), "looped": looped,
                     "new_tokens": answers[pid].get("new_tokens"), "max_new": prompt["max_new"]})
        fixable = [p for p in problems if "career: no 10th house" not in p and "discusses" not in p]
        if fixable:
            chosen, removed = remove_flagged_sentences(body, chart, stage)
            left = check_positions(chosen, chart) + check_analysis(chosen, chart, stage)
            if removed and not left and len(chosen.split()) >= 0.6 * len(body.split()):
                dpo.append({"id": pid, "prompt": prompt["messages"], "chosen": chosen, "rejected": body,
                            "removed": removed})
    out = bench / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out / "dpo_candidates.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in dpo) + "\n",
                                              encoding="utf-8")
    report = _report(rows, name, len(prompts), len(dpo))
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)


def _rate(rows) -> str:
    n = len(rows)
    if not n:
        return "| - | - | - | - |"
    clean = sum(r["n_problems"] == 0 for r in rows)
    per100 = 100 * sum(r["n_problems"] for r in rows) / sum(r["sentences"] for r in rows)
    return (f"| {n} | {clean / n:.0%} | {statistics.mean(r['n_problems'] for r in rows):.2f} | {per100:.1f} |")


def _report(rows, name, n_prompts, n_dpo) -> str:
    lines = [f"# Chart benchmark: {name}", "",
             f"{len(rows)} of {n_prompts} answers scored (first drafts, before the app's repair turn), as of {ON}.", "",
             "| Group | Answers | Error-free | Errors per answer | Errors per 100 sentences |", "|---|---|---|---|---|",
             "| **All** " + _rate(rows)]
    for task in ("reading", "question"):
        lines.append(f"| {task} " + _rate([r for r in rows if r["task"] == task]))
    for stage in ("child", "young person", "adult", "elder"):
        lines.append(f"| {stage} " + _rate([r for r in rows if r["stage"] == stage]))
    for flag in ("debilitated", "cancelled debility", "retrograde", "lagna near a cusp"):
        lines.append(f"| chart has: {flag} " + _rate([r for r in rows if flag in r["flags"]]))
    cats = Counter(category(p) for r in rows for p in r["problems"])
    lines += ["", "| Error type | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cats.most_common()]
    looped = sum(r["looped"] for r in rows)
    cut = sum(1 for r in rows if r["new_tokens"] and r["new_tokens"] >= r["max_new"])
    lines += ["", f"Answers that looped: {looped}. Answers cut at the length limit: {cut}. "
              f"Median length {statistics.median(r['words'] for r in rows):.0f} words. "
              f"Answers citing a passage: {sum(r['citations'] > 0 for r in rows)}.",
              f"DPO candidates (draft with errors; code removed them and nothing else was flagged): {n_dpo}."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- comparing two runs

def compare(bench: Path, a: str, b: str, iterations: int = 20000) -> None:
    load = lambda n: {r["id"]: r for r in map(json.loads, (bench / "runs" / n / "scores.jsonl").open(encoding="utf-8"))}
    ra, rb = load(a), load(b)
    ids = sorted(set(ra) & set(rb))
    diff = [rb[i]["n_problems"] - ra[i]["n_problems"] for i in ids]
    observed = statistics.mean(diff)
    rng = random.Random(1)
    extreme = sum(abs(statistics.mean(d if rng.random() < 0.5 else -d for d in diff)) >= abs(observed) - 1e-12
                  for _ in range(iterations))
    boots = sorted(statistics.mean(rng.choice(diff) for _ in diff) for _ in range(2000))
    a_only = sum(ra[i]["n_problems"] == 0 and rb[i]["n_problems"] > 0 for i in ids)
    b_only = sum(rb[i]["n_problems"] == 0 and ra[i]["n_problems"] > 0 for i in ids)
    print(f"{len(ids)} paired answers. Errors per answer: {a} {statistics.mean(ra[i]['n_problems'] for i in ids):.2f}, "
          f"{b} {statistics.mean(rb[i]['n_problems'] for i in ids):.2f}; difference {observed:+.2f} "
          f"(95% bootstrap interval {boots[50]:+.2f} to {boots[1949]:+.2f}); paired permutation p = "
          f"{(extreme + 1) / (iterations + 1):.4f}.")
    print(f"Error-free in {a} only: {a_only}; in {b} only: {b_only} (McNemar on these discordant pairs).")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scale", type=float, default=1.0, help="fraction of the charts per age band (smoke test)")
    p = sub.add_parser("score")
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--name", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("runs", nargs=2)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.scale)
    elif a.cmd == "score":
        score(a.bench, a.outputs, a.name)
    else:
        compare(a.bench, *a.runs)


if __name__ == "__main__":
    sys.exit(main())
