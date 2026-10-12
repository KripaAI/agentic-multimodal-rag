"""Phase 11: Gemini judges readings (with a fake judge and fake writers here)."""

from __future__ import annotations

import json

import pytest

from mmrag.astro import judge as J
from mmrag.astro import natal
from mmrag.astro.reading import Passage, Reading
from mmrag.config import PROJECT_ROOT, load_settings

needs_ephemeris = pytest.mark.skipif(not (natal._ephemeris_dir() / "de440s.bsp").is_file(),
                                     reason="JPL ephemeris not downloaded")


@pytest.fixture
def settings():
    return load_settings(PROJECT_ROOT / "config.yaml", {"DATABASE_URL": "postgresql://u:p@localhost:5432/mmrag"})


GOOD = {"errors": [{"quote": "Virgo is ruled by Mars", "problem": "Mercury rules Virgo", "severity": "major"}],
        "scores": {c: 4 for c in J.CRITERIA}, "overall": 4, "verdict": "Solid."}


def test_the_judges_json_is_read_even_inside_a_code_fence():
    text = "```json\n" + json.dumps(GOOD) + "\n```"
    j = J.Judgement.model_validate_json(J._json_in(text))
    assert j.overall == 4 and j.errors[0].severity == "major"


def test_the_cases_file_loads():
    cases = J.load_cases(PROJECT_ROOT / "eval" / "astro_cases.yaml")
    ids = [c["id"] for c in cases["readings"]]
    assert "child-delhi" in ids and "safety-death" in ids and len(cases["questions"]) >= 4


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        J.gemini_judge()


@needs_ephemeris
def test_run_writes_judges_and_reports(settings, tmp_path):
    cases = {"readings": [{"id": "child", "date": "2020-04-20", "time": "16:46", "tz": "Asia/Kolkata", "lat": 28.61,
                           "lon": 77.21, "question": "future", "on": "2026-10-12"}],
             "questions": [{"id": "q1", "question": "What is Gajakesari yoga?"}]}
    seen = []

    def reading_fn(s, chart, question=None, writer="astro", use_books=True, on=None):
        p = [Passage(1, "c1", "book.pdf", [3], "text", "q")]
        return Reading(text="A reading [1].", model=f"{writer}-model", passages=p, cited=p, notices=["a note"],
                       chart=chart, facts="FACTS")

    def answer_fn(s, question, writer="astro", use_books=True):
        return Reading(text="An answer.", model=f"{writer}-model", passages=[], cited=[], notices=[])

    def judge(user: str) -> J.Judgement:
        seen.append(user)
        return J.Judgement.model_validate(GOOD)

    report = J.run(settings, cases, ["astro", "openai"], judge, "fake-judge", reading_fn=reading_fn,
                   answer_fn=answer_fn)
    assert len(report.results) == 4 and all(r.judgement for r in report.results)
    assert "CHART FACTS:\nFACTS" in seen[0] and "[P1] (book.pdf, page 3) text" in seen[0]
    assert "REQUEST:\nfuture" in seen[0] and "READING:\nA reading [1]." in seen[0]
    assert "(none: a general question)" in seen[2]
    s = J.summary(report)
    assert s["astro"]["overall"] == 4 and s["astro"]["major_errors"] == 1 and s["openai"]["cases"] == 2
    path = J.write_report(report, tmp_path)
    md = path.read_text(encoding="utf-8")
    assert "| astro | 2 | 4.0 |" in md and "Virgo is ruled by Mars" in md
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["judge_model"] == "fake-judge"


def test_a_failing_case_does_not_stop_the_run(settings):
    cases = {"questions": [{"id": "q1", "question": "x"}, {"id": "q2", "question": "y"}]}

    def answer_fn(s, question, writer="astro", use_books=True):
        if question == "x":
            raise TimeoutError("server asleep")
        return Reading(text="ok", model="m", passages=[], cited=[], notices=[])

    report = J.run(settings, cases, ["astro"], lambda u: J.Judgement.model_validate(GOOD), "j", answer_fn=answer_fn)
    assert report.results[0].error.startswith("TimeoutError") and report.results[1].judgement


def test_a_judge_failure_keeps_the_reading_and_rejudge_finishes_it(settings, tmp_path):
    cases = {"questions": [{"id": "q1", "question": "x"}]}
    p = [Passage(1, "c1", "book.pdf", [3], "text", "q")]

    def answer_fn(s, question, writer="astro", use_books=True):
        return Reading(text="An answer [1].", model="m", passages=p, cited=p, notices=[])

    def quota(user):
        raise RuntimeError("429 quota")

    report = J.run(settings, cases, ["astro"], quota, "j", answer_fn=answer_fn)
    r = report.results[0]
    assert r.reading == "An answer [1]." and r.judgement is None and r.error.startswith("judge: RuntimeError")
    J.write_report(report, tmp_path)
    assert "Not judged" in (tmp_path / "report.md").read_text(encoding="utf-8")
    seen = []
    again = J.rejudge(J.load_report(tmp_path / "results.json"), lambda u: seen.append(u) or J.Judgement.model_validate(GOOD),
                      "j2")
    assert again.results[0].judgement["overall"] == 4 and again.results[0].error is None
    assert "[P1] (book.pdf, page 3) text" in seen[0] and again.judge_model == "j2"
    written_only = J.run(settings, cases, ["astro"], None, "j", answer_fn=answer_fn)
    assert written_only.results[0].judgement is None and written_only.results[0].error is None


@needs_ephemeris
def test_code_written_sections_and_reference_facts():
    from datetime import date

    from mmrag.astro.compose import collapse_repetition, factual_sections, reference_facts, strip_factual_sections
    from mmrag.astro.natal import parse_birth

    child = natal.compute_chart(parse_birth("2020-04-20", "16:46", "Asia/Kolkata", 28.61, 77.21))
    text = factual_sections(child, date(2026, 10, 12))
    assert "Virgo Lagna" in text and "the 5th (Capricorn): Mars, Jupiter and Saturn" in text
    assert "Challenge: Mercury is debilitated in Pisces and the debility is not cancelled" in text
    assert "Challenge: Mars is debilitated in the Navamsa (Cancer)" in text and "longevity" not in text
    assert "Ashlesha, Jyeshtha and Revati" in reference_facts("Which nakshatras are ruled by Mercury?")
    assert "Jupiter in a kendra" in reference_facts("What is Gajakesari yoga?")
    assert "Hasta nakshatra is ruled by Moon" in reference_facts("Tell me about Hasta")
    assert reference_facts("hello") == ""
    looped, cut = collapse_repetition("a line of text\n" + "- the Moon is not in the 5th\n" * 5)
    assert cut and looped.count("the Moon is not") == 2
    assert strip_factual_sections("**Relevant factors**\n- x\n\n**Summary**\nok") == "**Summary**\nok"


def test_case_dates_survive_yaml():
    """A bare `on:` key is the boolean True in YAML; the Gandhi case silently ran on today's date."""
    cases = J.load_cases(PROJECT_ROOT / "eval" / "astro_cases.yaml")
    gandhi = next(c for c in cases["readings"] if c["id"] == "historical-gandhi")
    assert gandhi["as_of"] == "1930-03-12" and True not in gandhi


def test_book_names_are_cited_after_the_whole_name():
    from mmrag.astro.reading import resolve_citations

    ps = [Passage(1, "c1", "bhrigu-sutras.pdf", [3], "Bhrigu text", "q"),
          Passage(2, "c2", "saturn-friend-and-guide.pdf", [5], "Saturn-Friend & Guide text", "q")]
    out = resolve_citations("The Bhrigu Sutras describe it. The Saturn-Friend & Guide view differs.", ps)[0]
    assert out == "The Bhrigu Sutras [1] describe it. The Saturn-Friend & Guide [2] view differs."
