"""RAGAS-style judge metrics with a scripted judge (no API calls). Test-first."""

from __future__ import annotations

import pytest

from mmrag.eval.judge import (Claim, ClaimVerdicts, ContextVerdicts, FactualVerdicts, GeneratedQuestions,
                              Metrics, Verdict)

pytestmark = pytest.mark.unit


class FakeJudge:
    def __init__(self, *replies, vectors=None):
        self.replies, self.vectors, self.calls = list(replies), vectors or {}, []

    def structured(self, messages, schema):
        self.calls.append((schema.__name__, messages))
        reply = self.replies.pop(0)
        assert isinstance(reply, schema)
        return reply

    def embed(self, texts):
        return [self.vectors[t] for t in texts]


CTX = [("d:text:1", "GQA shrinks the KV cache from 33.5 GB to 6.7 GB."), ("d:text:2", "Unrelated passage.")]


def test_faithfulness_is_the_share_of_supported_claims_and_sees_the_evidence():
    judge = FakeJudge(ClaimVerdicts(claims=[Claim(statement="GQA shrinks the cache", supported=True, reason="p1"),
                                            Claim(statement="It is 10x smaller", supported=False, reason="5x")]))
    score, details = Metrics(judge).faithfulness("q?", "GQA shrinks the cache 10x.", CTX)
    assert score == 0.5 and details["claims"][1]["reason"] == "5x"
    prompt = judge.calls[0][1][-1]["content"]
    assert "33.5 GB" in str(prompt) and "d:text:1" in str(prompt)


def test_multimodal_faithfulness_sends_the_figure_images():
    judge = FakeJudge(ClaimVerdicts(claims=[Claim(statement="s", supported=True, reason="")]))
    Metrics(judge).faithfulness("q?", "a", CTX, images=["data/assets/x.png"])
    parts = judge.calls[0][1][-1]["content"]
    assert {"type": "image_path", "path": "data/assets/x.png"} in parts


def test_no_claims_means_not_applicable():
    assert Metrics(FakeJudge(ClaimVerdicts(claims=[]))).faithfulness("q", "a", CTX)[0] is None


def test_response_relevancy_uses_generated_questions_and_embeddings():
    judge = FakeJudge(GeneratedQuestions(questions=["a", "b"], noncommittal=False),
                      vectors={"q": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert Metrics(judge).response_relevancy("q", "answer")[0] == pytest.approx(0.5)
    evasive = FakeJudge(GeneratedQuestions(questions=["a"], noncommittal=True), vectors={"q": [1, 0], "a": [1, 0]})
    assert Metrics(evasive).response_relevancy("q", "I am not sure")[0] == 0.0


def test_context_precision_is_average_precision_in_retrieval_order():
    judge = FakeJudge(ContextVerdicts(verdicts=[Verdict(useful=False, reason=""), Verdict(useful=True, reason=""),
                                                Verdict(useful=True, reason="")]))
    ctx = CTX + [("d:text:3", "x")]
    # relevant at ranks 2 and 3: (1/2 + 2/3) / 2
    assert Metrics(judge).context_precision("q", "ref", ctx)[0] == pytest.approx((0.5 + 2 / 3) / 2)


def test_context_recall_is_the_share_of_reference_statements_found():
    judge = FakeJudge(ClaimVerdicts(claims=[Claim(statement="a", supported=True, reason=""),
                                            Claim(statement="b", supported=True, reason=""),
                                            Claim(statement="c", supported=False, reason="")]))
    assert Metrics(judge).context_recall("q", "ref", CTX)[0] == pytest.approx(2 / 3)


def test_factual_correctness_is_claim_f1():
    judge = FakeJudge(FactualVerdicts(answer_claims=[Claim(statement="a", supported=True, reason=""),
                                                     Claim(statement="b", supported=False, reason="")],
                                      reference_claims=[Claim(statement="r", supported=True, reason="")]))
    # precision 1/2, recall 1/1 -> F1 2/3
    assert Metrics(judge).factual_correctness("answer", "reference")[0] == pytest.approx(2 / 3)
