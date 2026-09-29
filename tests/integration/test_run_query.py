"""run_query end to end on a real database with a scripted model: checkpoints, query_log,
threads, the answer page. No OpenAI calls. Written test-first."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.agent.graph import PlanOut, run_query
from mmrag.agent.answer import Answer
from mmrag.agent.llm import LLMReply
from mmrag.agent.render import write_answer_page
from mmrag.config import load_settings
from mmrag.index.chunk import Chunk
from mmrag.index.document import LoadedDoc
from mmrag.index.writer import write_document
from mmrag.ingest.models import Element
from mmrag.obs.querylog import cost_usd

pytestmark = pytest.mark.integration
DOC = "f" * 16


class ScriptedLLM:
    model = "scripted"

    def __init__(self):
        self.searched = 0

    def chat(self, messages, tools_spec):
        if self.searched == 0:
            self.searched += 1
            return LLMReply(None, [{"id": "c1", "name": "search_text", "arguments": {"query": "KV cache"}}], 100, 10)
        self.searched = 0
        return LLMReply("enough", [], 120, 5)

    def structured(self, messages, schema, tools_spec=None):
        if schema is PlanOut:
            return PlanOut(qtype="conceptual", note="text"), LLMReply(None, [], 50, 5)
        return Answer.model_validate({"blocks": [{"type": "text", "markdown": "GQA shrinks the **KV cache**.",
                                                  "citations": [{"id": f"{DOC}:text:1"}]}]}), LLMReply(None, [], 300, 40)


@pytest.fixture
def settings(fresh_db_url, base_config, write_config, tmp_path):
    base_config["paths"]["data_dir"] = str(tmp_path / "data")
    base_config["observability"].update(enabled=False, log_file=None)
    base_config["pricing"] = {"scripted": {"input_per_mtok": 1.0, "output_per_mtok": 4.0}}
    s = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    el = Element(element_id=f"{DOC}:p19:text:1", doc_id=DOC, source_file="t.pdf", page=19, bbox=(10, 20, 300, 60),
                 type="text", text="GQA shrinks the KV cache from 33.5 GB to 6.7 GB.", content_hash="t")
    doc = LoadedDoc(doc_id=DOC, source_file="t.pdf", content_hash="x", elements=[el])
    chunk = Chunk(chunk_id=f"{DOC}:text:1", collection="text", element_ids=[el.element_id], dense_text=el.text,
                  keyword_text=el.text)
    write_document(s, doc, [chunk], [[1.0] + [0.0] * (s.embed.dims - 1)], [])
    return s


def _embed(settings):
    return lambda q: [1.0] + [0.0] * (settings.embed.dims - 1)


def test_cost_from_the_pricing_table(settings):
    assert cost_usd("scripted", 1_000_000, 500_000, settings) == pytest.approx(3.0)
    assert cost_usd("unpriced-model", 1000, 1000, settings) is None


def test_run_query_logs_hydrated_answer_and_continues_the_thread(settings):
    run = run_query("How much does GQA save?", settings, llm=ScriptedLLM(), embed_query=_embed(settings))
    loc = run.answer.blocks[0].citations[0].locations[0]
    assert (loc.source_file, loc.page, loc.bbox) == ("t.pdf", 19, (10.0, 20.0, 300.0, 60.0))  # from the database
    assert run.rounds == 1 and run.validator_result == "ok" and len(run.trace_id) == 32
    assert run.input_tokens == 50 + 100 + 120 + 300 and run.cost_usd == pytest.approx((570 * 1 + 60 * 4) / 1e6)

    follow = run_query("And per token?", settings, thread_id=run.thread_id, llm=ScriptedLLM(),
                       embed_query=_embed(settings))
    assert follow.thread_id == run.thread_id
    with db.connect(settings) as conn:
        rows = conn.execute("SELECT thread_id, question, rounds, tokens_in, cost_usd, validator_result "
                            "FROM query_log ORDER BY created_at").fetchall()
        saved = conn.execute("SELECT count(DISTINCT thread_id) FROM checkpoints").fetchone()[0]
    assert [r[1] for r in rows] == ["How much does GQA save?", "And per token?"]
    assert {r[0] for r in rows} == {run.thread_id} and rows[0][3] == 570 and rows[0][5] == "ok"
    assert saved == 1  # one conversation, checkpointed in PostgreSQL


def test_answer_page_shows_blocks_sources_and_run_details(settings):
    run = run_query("How much does GQA save?", settings, llm=ScriptedLLM(), embed_query=_embed(settings))
    page = write_answer_page("How much does GQA save?", run, settings)
    html = page.read_text(encoding="utf-8")
    assert page.parent == settings.resolve(settings.paths.data_dir) / "answers"
    assert "<strong>KV cache</strong>" in html  # markdown rendered
    assert "t.pdf" in html and "p. 19" in html and run.trace_id in html and run.thread_id in html
