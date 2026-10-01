# Phase 6 — Evaluation results

**Date:** 2026-09-30 · **Branch:** `phase-6` · **Golden set:** `eval/golden_set.jsonl` (40 questions, version `cba0847e4970`)

## 1. Setup

- **Golden set.** 40 questions drafted from all five PDFs by the judge model, then reviewed and edited by the owner: 15 conceptual, 10 visual, 8 quantitative, 4 mixed, 3 unanswerable. Every answerable question has reference evidence IDs.
- **Grader.** Our own RAGAS-style judge (`src/mmrag/eval/judge.py`), because installing RAGAS would downgrade `openai` 3.x to 1.x (owner decision). The metric definitions follow RAGAS: faithfulness (also against figure images), response relevancy, context precision and recall, factual correctness. There are also deterministic checks: chart values, figure shown, citations, refusal, tool calls, latency and cost.
- **Judge model.** `gpt-4.1-mini`, at temperature 0 with a fixed seed. The owner chose it over `gpt-4.1` for budget; it costs about 5× less.
- **Agent.** `gpt-5.4-mini`, production configuration.
- **Budget.** The owner capped Phase 6 at **US$3** after adding credits. `eval run --max-cost` enforces the cap per run.

## 2. Baseline (production configuration)

| Metric | Baseline | v1 target (spec §9) | |
|---|---|---|---|
| Faithfulness | **1.00** | ≥ 0.90 | ✅ |
| Multimodal faithfulness | **1.00** | ≥ 0.85 | ✅ |
| Context recall | **0.99** | ≥ 0.80 | ✅ |
| Context precision | **0.88** | ≥ 0.80 | ✅ |
| Tool call accuracy | **0.93** | ≥ 0.80 | ✅ |
| Figure hit rate | **0.86** | ≥ 0.80 | ✅ |
| Correct refusal on unanswerable | **3/3** | 3/3 | ✅ |
| Factual correctness | 0.74 | ≥ 0.75 | ≈ (see §5) |
| Citation accuracy | 0.67 | ≥ 0.90 | ❌ (see §5) |
| Response relevancy | 0.71 | ≥ 0.85 | ❌ calibration (see §5) |
| Chart numeric correctness | 0.63* | 100% | ❌ (see §4, §5) |
| Median / p90 latency | **9.3 s / 13.8 s** | ≤ 20 s / ≤ 45 s | ✅ |
| Median cost per question | **US$0.014** | ≤ US$0.15 | ✅ |

\* 0.50 as first graded. One correct chart was misread by the grader (`1.67772e+07`); that bug is fixed.

**Reading:** answers are grounded, the needed evidence is almost always found, figures are shown, and the system refuses honestly. It is fast and cheap. The weak spot is charts.

## 3. Experiments and decisions (P10: keep only what measurably helps)

| Change | How it was measured | Result | Decision |
|---|---|---|---|
| Local cross-encoder reranker (MiniLM-L6, fastembed) | Search-only on golden references (cents) | Top-5 0.92 → 0.97, MRR 0.86 → 0.93 | Built, **off**: no gain end to end (below) |
| Keyword "any word" | Search-only | Worse alone (MRR 0.75); helps only with the reranker | Built, **off** |
| BM25 (ParadeDB `pg_search`) | Search-only, on a throwaway ParadeDB copy | Text top-5 0.63 → 0.74, but MRR no better than the reranker | **Not adopted.** It would mean a database change for a gain the reranker already covers. |
| Candidate count, RRF k, HNSW `ef_search` | Search-only | No change | Keep defaults |
| Reranker + keyword any + prompts v2 + 4 rounds | Full 40-question run | Worse on almost every metric; latency 9.3 → 13.0 s | **Rejected.** Retrieval wasn't the bottleneck (recall 0.99). |
| Prompts v3 (v1 + chart rules only) + 4 rounds | The 8 chart questions | Chart accuracy 0.63 → **0.88**. Other scores lower, mostly from one question (n11) whose data wasn't found on this run | **Candidate.** Confirming it needs a full run (about US$1, over budget). Switch with `agent.prompt_version: v3`. |
| Round limit 3 vs 4 vs 5 | Query-log analysis (free) | 24 of the 25 answers that used all 3 rounds were chart questions; other types rarely use more than 1 | Keep 3. Revisit together with v3. |

## 4. Bugs found by the evaluation (all fixed, with tests)

- Chart data tables rounded large values (16,777,216 was shown as `1.67772e+07`). That was a display truthfulness bug (P4).
- One shared trace for every question inside `eval.run`. Each question now has its own trace.
- Out-of-credits errors were retried as rate limits for hours. They now stop the run immediately.
- Windows standby froze the run for up to 36 minutes per question. `eval run` now keeps the machine awake.
- A damaged line in the embedding cache (from two concurrent writers) crashed every query. Damaged lines are now skipped.
- Grader fixes:
  - context precision is scored per search call;
  - citations on the same page as the reference count;
  - charts are "not applicable" when none is expected;
  - a question that fails to run now fails the gate.

## 5. Gaps for the owner to accept (acceptance criterion: meet targets or accept documented gaps)

1. **Chart numeric correctness below 100%.** Charts fail when the agent charts a midpoint of a printed range, mixes units, or doesn't find the table (n11). The chart engine rejects every unverified value, so **no wrong chart is ever shown**. The miss is a missing chart, not a false one. Prompts v3 raise this to 0.88 on the chart questions.
2. **Citation accuracy 0.67.** Answers often cite correct passages that aren't in the golden set's single reference passage. The metric is stricter than the truth it measures. Proposal: set the target to ≥ 0.70 until references list all supporting passages.
3. **Response relevancy 0.71.** Embedding cosine similarity between paraphrased questions rarely exceeds about 0.8 with `text-embedding-3-large`, even for on-topic answers. Proposal: calibrate the target to ≥ 0.70.
4. **Factual correctness 0.74 (target 0.75).** In the spot-check below, every "unsupported" verdict was a *true* statement missing from the short reference answer. The metric partly penalizes correct extra detail. Faithfulness (1.00) is the reliable truth measure.
5. **Judge variance not measured.** The second baseline run was skipped for budget. Differences under about 0.05 between runs should be treated as noise.

## 6. Owner spot-check: 10 judge verdicts from the baseline

The full list, with reasons, is in `data/eval/spot_check.json` (local file).

| # | Question | Metric | Statement (abridged) | Judge |
|---|---|---|---|---|
| 1 | v05 | factual correctness | Packed examples need document-aware attention and loss masking | not in reference |
| 2 | m05 | factual correctness | The method stack is SFT, DPO/SimPO, … | not in reference |
| 3 | m04 | factual correctness | Qwen3-14B has 40 query heads and 8 KV heads | not in reference |
| 4 | c11 | factual correctness | DPO optimizes against a frozen reference policy | not in reference |
| 5 | v12 | factual correctness | Text outputs are normalized into a shared representation | not in reference |
| 6 | n10 | faithfulness | 336 × 336 → 576 tokens per image | supported |
| 7 | v11 | faithfulness | The original fix is a second lookup table per position | supported |
| 8 | n11 | factual correctness | Late interaction needs 10–100× more storage | supported |
| 9 | n05 | factual correctness | T5-Small gives a 3.4× speedup | supported |
| 10 | c05 | faithfulness | PagedAttention reduces fragmentation | supported |

## 7. Final configuration

- Search: hybrid (semantic + all-words keyword, RRF), no reranker. The reranker, keyword modes and BM25 remain available as settings.
- Agent: `gpt-5.4-mini`, **prompts `v3` and 4 rounds (5 for multi-part)**. The owner adopted v3 on 2026-09-30, in the configuration tested on the chart questions (chart accuracy 0.63 → 0.88). A live check on n03 produced an accepted chart of printed values (20 %, 60 %) instead of the rejected midpoint 67.5. A full 40-question confirmation run is still pending budget.
- Evaluation: judge `gpt-4.1-mini`, `mmrag eval run --max-cost`, and the baseline run recorded in `eval_runs`.

## 8. Spend

| Item | US$ |
|---|---|
| Before the credit top-up (drafting, smoke tests, first baseline attempt with `gpt-4.1` judge) | ≈ 3.30 |
| After the top-up: baseline (40 questions) | 0.92 |
| After the top-up: combined "improved" run (40 questions) | 1.02 |
| After the top-up: chart-question run with prompts v3 (8 questions) | 0.29 |
| **After the top-up, total** | **≈ 2.23 of the US$3 cap** |

---

## 9. Memory on vs the baseline (Phase 9, acceptance criterion 5)

**The claim to test:** switching long-term memory on does not lower faithfulness or citation accuracy. Memory personalises; it is never evidence (P13), so the scores should be flat.

**Why the baseline is unaffected.** `mmrag eval run` passes no user, so memory is never recalled or stored in a normal run — the Phase 6 baseline above stands as the memory-off measurement.

**Procedure** (about US$1 per 40-question run at the prices in §8, so budget for two):

```powershell
mmrag user add eval-memory@example.com           # a throwaway account, so no real user's memory is polluted
mmrag memory add output_format "Prefers charts to tables." eval-memory@example.com
mmrag memory add topic_focus "Is studying attention and KV caching." eval-memory@example.com
mmrag eval run --as-user eval-memory@example.com --max-cost 1.50
mmrag user delete eval-memory@example.com --yes  # removes the account and its memories
```

The notes are seeded by hand because the golden set asks about documents: a run that only stores (`--warmup`) may legitimately extract nothing, leaving nothing to measure. `--as-user` is refused together with `--baseline`: the baseline must stay memory-off. Note that the run also *adds* memories as it goes, so a repeat run is not identical — delete and re-seed the account to repeat it exactly. The report's **vs baseline** column shows Δ per metric, and the regression gate fails the run if a gated metric is more than `eval.regression_tolerance` (0.03) below the baseline — so a pass *is* the criterion met.

**Result (2026-10-01, run `c9fd5874`, commit `c677716`, label "memory on"):** 39 of 40 questions; the run stopped at a US$0.85 cost cap one question short of the end (`u3`, an unanswerable question — `refusal` is therefore 2/2, not 3/3).

| Metric | Baseline `a147d95f` | Memory on `c9fd5874` | Δ | Within 0.03 |
|---|---|---|---|---|
| **Faithfulness** | 0.996 | 0.982 | −0.014 | ✅ |
| **Citation accuracy** | 0.675 | 0.676 | +0.002 | ✅ |
| Multimodal faithfulness | 1.000 | 1.000 | 0.000 | ✅ |
| Context precision | 0.876 | 0.871 | −0.005 | ✅ |
| Context recall | 0.993 | 0.973 | −0.020 | ✅ |
| Factual correctness | 0.741 | 0.739 | −0.001 | ✅ |
| Figure hit rate | 0.857 | 0.857 | 0.000 | ✅ |
| Response relevancy | 0.707 | 0.683 | −0.024 | ✅ |
| Correct refusal | 1.000 | 1.000 (2/2) | 0.000 | ✅ |
| Chart numeric | 0.500 | 0.625 | +0.125 | ✅ (still below the 1.0 target, §5) |
| **Tool call accuracy** | 0.932 | 0.865 | **−0.068** | ❌ beyond tolerance |
| Median cost per question | US$0.0139 | US$0.0130 | −0.0009 | — |
| Median latency | 9.3 s | 8.2 s | −1.1 s | — |

**The criterion is met: faithfulness and citation accuracy do not drop.** Both are flat, which is what the design predicts — memories reach the model as labelled user context and the validator refuses any citation to one, so there is no path by which a memory can become a claim. The `live` test `test_no_answer_ever_cites_a_memory` checks that path against the real agent.

**The gate still failed, for three reasons, and only one of them is about quality:**
1. the cost cap stopped the run one question early (budget, not a defect);
2. `chart_numeric` 0.625 is below its 1.0 target — a **pre-existing** gap (§4, §5), and better than the baseline's 0.500;
3. `tool_call_accuracy` fell 0.068, beyond the 0.03 tolerance. See the confound below.

**Important confound — this is not a clean memory-on/memory-off comparison.** The baseline was run on 2026-09-30 with agent prompts **v1 and 3 rounds**; this run used the adopted configuration, **v3 and 4 rounds** (§7, where a full 40-question confirmation run was recorded as "pending budget"). So this single run is *both* the pending v3 confirmation and the memory-on measurement, and its deltas mix the two changes. The two metrics that moved most are the ones v3 was designed to move: `chart_numeric` up (v3 is the chart-rules prompt) and `tool_call_accuracy` down (more chart and table fetches that the golden set's `expected_tool_calls` do not list — the seven questions below 1.0 are mixed and numeric ones calling extra searches, not memory-driven behaviour).

**To isolate memory** costs one more run: the same 40 questions at the current config with no `--as-user`, recorded as the new baseline, then this run compared against it. About US$0.9.

**The extractor stored nothing new across the whole run.** After 39 answered questions the account still held exactly the two seeded notes. That is the designed behaviour — the golden set asks about documents, and the prompt says an exchange that reveals nothing lasting about the user should produce no statement — and it confirms two things: the extractor is not over-eager (the main P13 risk), and the plan's warm-up idea could not have produced the notes this comparison needed, which is why `mmrag memory add` exists.

**Known gap in cost accounting.** `recall_memory` and `remember` do not appear in `query_log.cost_usd`: the extractor is a separate model client whose tokens never enter the agent's state. So the median cost above understates memory-on cost by roughly US$0.001 per answer (about US$0.04 across this run), and the `auth.daily_cost_limit_usd` per-user limit does not see it either. Not a correctness problem; worth fixing before memory is relied on for cost control.

## 10. Spend on the Phase 9 measurement

| Item | US$ |
|---|---|
| `pytest -m live`, two runs (6 agent questions) | 0.11 |
| Memory-on golden-set run, 39/40 questions (agent + judge) | 0.855 |
| Memory extraction calls, not counted above (estimated) | ≈ 0.04 |
| **Total** | **≈ 1.01 of the US$1 authorised** |
