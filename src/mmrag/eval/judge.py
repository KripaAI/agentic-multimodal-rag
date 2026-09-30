"""Model-judged metrics, RAGAS-style (owner decision, Phase 6: RAGAS itself would downgrade openai 3.x).

Each metric follows the published RAGAS definition, with one structured-output call to the fixed
judge model (D14) instead of RAGAS's multi-step prompts:

- faithfulness: answer claims supported by the retrieved evidence (images added for figures)
- response relevancy: mean cosine between the question and questions generated from the answer
- context precision (with reference): average precision of useful evidence in retrieval order
- context recall: reference statements attributable to the evidence
- factual correctness: claim-level F1 between answer and reference

Every score returns its details (claims, verdicts, reasons) for the report and the owner's spot-check.
"""

from __future__ import annotations

import math
import statistics
from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

JUDGE_PROMPT_VERSION = "judge-v1"
MAX_CONTEXT_CHARS = 1500
T = TypeVar("T", bound=BaseModel)


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(_M):
    statement: str
    supported: bool
    reason: str


class ClaimVerdicts(_M):
    claims: list[Claim]


class Verdict(_M):
    useful: bool
    reason: str


class ContextVerdicts(_M):
    verdicts: list[Verdict]  # one per context, in the order given


class GeneratedQuestions(_M):
    questions: list[str]
    noncommittal: bool  # the answer is evasive or says it does not know


class FactualVerdicts(_M):
    answer_claims: list[Claim]  # supported = backed by the reference
    reference_claims: list[Claim]  # supported = covered by the answer


class Judge(Protocol):
    def structured(self, messages: list[dict], schema: type[T]) -> T: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


SYSTEM = ("You are a strict, impartial evaluator of answers produced by a document question-answering system. "
          "Judge only from the material given in the message; never use outside knowledge. Keep reasons short.")

Context = tuple[str, str]  # (evidence id, text)


def _contexts(contexts: list[Context]) -> str:
    return "\n\n".join(f"[{i + 1}] ({cid})\n{text[:MAX_CONTEXT_CHARS]}" for i, (cid, text) in enumerate(contexts))


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def _average_precision(useful: list[bool]) -> float | None:
    """RAGAS context precision for one ranked list; None when nothing in it was useful."""
    hits, total = 0, 0.0
    for k, u in enumerate(useful, 1):
        if u:
            hits += 1
            total += hits / k
    return total / hits if hits else None


def _share(claims: list[Claim]) -> float | None:
    return sum(c.supported for c in claims) / len(claims) if claims else None


class Metrics:
    def __init__(self, judge: Judge):
        self.judge = judge

    def _ask(self, text: str, schema: type[T], images: list[str] | None = None) -> T:
        content = [{"type": "text", "text": text}, *({"type": "image_path", "path": p} for p in images)] \
            if images else text
        return self.judge.structured([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                                     schema)

    def faithfulness(self, question: str, answer: str, contexts: list[Context], images: list[str] | None = None
                     ) -> tuple[float | None, dict]:
        out = self._ask(
            "Break the ANSWER into short, self-contained factual statements (skip greetings and statements "
            "that only say something could not be found). For each, set supported=true only if it can be "
            "directly inferred from the EVIDENCE" + (" or the attached figure images" if images else "")
            + f".\n\nQUESTION: {question}\n\nANSWER:\n{answer}\n\nEVIDENCE:\n{_contexts(contexts)}",
            ClaimVerdicts, images)
        return _share(out.claims), {"claims": [c.model_dump() for c in out.claims]}

    def response_relevancy(self, question: str, answer: str, n: int = 3) -> tuple[float | None, dict]:
        out = self._ask(
            f"Write {n} different questions that the ANSWER below answers directly. Set noncommittal=true if the "
            f"answer is evasive, vague or says it does not know.\n\nANSWER:\n{answer}", GeneratedQuestions)
        if out.noncommittal:
            return 0.0, {"questions": out.questions, "noncommittal": True}
        if not out.questions:
            return None, {}
        vectors = self.judge.embed([question, *out.questions])
        score = sum(_cosine(vectors[0], v) for v in vectors[1:]) / len(out.questions)
        return score, {"questions": out.questions}

    def context_precision(self, question: str, reference: str, contexts: list[Context],
                          rankings: list[list[str]] | None = None) -> tuple[float | None, dict]:
        """With `rankings` (the result ids of each search call, in rank order), each search is scored
        on its own and the searches that found something useful are averaged: an agent's parallel
        searches have no meaningful order between them. A search that found nothing is left out;
        0 when none did."""
        if not contexts:
            return None, {}
        out = self._ask(
            f"For each of the {len(contexts)} numbered EVIDENCE items, in order, decide whether it was useful for "
            f"arriving at the REFERENCE ANSWER to the QUESTION. Return exactly one verdict per item.\n\n"
            f"QUESTION: {question}\n\nREFERENCE ANSWER:\n{reference}\n\nEVIDENCE:\n{_contexts(contexts)}",
            ContextVerdicts)
        useful = [v.useful for v in out.verdicts[:len(contexts)]]
        details = {"verdicts": [v.model_dump() for v in out.verdicts]}
        if not rankings:
            ap = _average_precision(useful)
            return (ap if ap is not None else 0.0), details
        by_id = {cid: u for (cid, _), u in zip(contexts, useful)}
        per = [_average_precision([by_id.get(i, False) for i in r]) for r in rankings]
        found = [p for p in per if p is not None]
        details["per_search"] = per
        return (statistics.fmean(found) if found else 0.0), details

    def context_recall(self, question: str, reference: str, contexts: list[Context]) -> tuple[float | None, dict]:
        out = self._ask(
            "Break the REFERENCE ANSWER into short factual statements. For each, set supported=true if it can be "
            "attributed to the EVIDENCE.\n\n"
            f"QUESTION: {question}\n\nREFERENCE ANSWER:\n{reference}\n\nEVIDENCE:\n{_contexts(contexts)}",
            ClaimVerdicts)
        return _share(out.claims), {"claims": [c.model_dump() for c in out.claims]}

    def factual_correctness(self, answer: str, reference: str) -> tuple[float | None, dict]:
        out = self._ask(
            "Break the ANSWER and the REFERENCE into short factual statements. For each answer statement, set "
            "supported=true if the REFERENCE backs it. For each reference statement, set supported=true if the "
            f"ANSWER states it.\n\nANSWER:\n{answer}\n\nREFERENCE:\n{reference}", FactualVerdicts)
        precision, recall = _share(out.answer_claims), _share(out.reference_claims)
        if precision is None or recall is None:
            return None, {}
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return f1, {"precision": precision, "recall": recall, "answer_claims": [c.model_dump() for c in out.answer_claims],
                    "reference_claims": [c.model_dump() for c in out.reference_claims]}


class OpenAIJudge:
    """The fixed judge (eval.judge_model) through our OpenAI client; temperature 0 and a fixed seed."""

    def __init__(self, client, model: str, embed_model: str, dims: int):
        self.client, self.model, self.embed_model, self.dims = client, model, embed_model, dims

    def structured(self, messages: list[dict], schema: type[T]) -> T:
        from openai.lib._parsing._completions import type_to_response_format_param

        from mmrag.agent.llm import _wire
        from mmrag.llm import REASONING_PREFIXES

        extra = {"reasoning_effort": "low"} if self.model.startswith(REASONING_PREFIXES) else {"temperature": 0}
        r = self.client.chat.completions.create(model=self.model, messages=_wire(messages), seed=7,
                                                response_format=type_to_response_format_param(schema),
                                                max_completion_tokens=4000, **extra)
        self.usage = getattr(self, "usage", [0, 0])
        self.usage[0] += r.usage.prompt_tokens
        self.usage[1] += r.usage.completion_tokens
        return schema.model_validate_json(r.choices[0].message.content)

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.embeddings.create(model=self.embed_model, input=texts, dimensions=self.dims)
        return [d.embedding for d in r.data]
