"""UI smoke test (plan Phase 7 task 11): the real Streamlit script, run headless against a
throwaway database. No API calls: stored answers are rendered, the agent is never run."""

from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import mmrag.db as db
from mmrag.auth import service
from mmrag.config import PROJECT_ROOT, load_settings

pytestmark = pytest.mark.integration
APP = str(PROJECT_ROOT / "app" / "ui.py")
PW = "correct horse battery staple"

SAMPLE_ANSWER = {  # the shape of a Phase 4 hydrated answer: text, chart with data table, table
    "blocks": [
        {"type": "text", "content": {"markdown": "GQA shrinks the **KV cache**."},
         "citations": [{"id": "d:text:1", "locations": [{"element_id": "d:p19:text:1", "source_file": "t.pdf",
                                                          "page": 19, "bbox": [10, 20, 300, 60]}]}]},
        {"type": "chart", "approximate": False, "citations": [],
         "content": {"chart_id": "chart-1", "chart_type": "bar", "title": "KV", "notes": [],
                     "spec": {"data": [{"type": "bar", "x": ["40 heads", "8 heads"], "y": [800, 160]}],
                              "layout": {"title": {"text": "KV"}}},
                     "data_table": [["", "KB"], ["40 heads", "800"], ["8 heads", "160"]]}},
        {"type": "table", "citations": [], "content": {"title": "Memory", "columns": ["", "40", "8"],
                                                      "rows": [["per token", "800 KB", "160 KB"]]}}],
    "sources": [{"source_file": "t.pdf", "page": 19, "element_id": "d:p19:text:1"}], "notices": []}


@pytest.fixture
def settings(fresh_db_url, base_config, write_config, monkeypatch, tmp_path):
    common = tmp_path / "common.txt"
    common.write_text("password1234\n", encoding="utf-8")
    base_config["auth"]["common_passwords_file"] = str(common)
    base_config["observability"].update(enabled=False, log_file=None)
    path = write_config(base_config)
    s = load_settings(path, {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    monkeypatch.setenv("MMRAG_CONFIG", str(path))
    monkeypatch.setenv("DATABASE_URL", fresh_db_url)
    st.cache_resource.clear()  # the app caches settings per process
    st.cache_data.clear()
    return s


def _signed_in(settings, role="user"):
    temp = service.add_user(settings, "dana@example.com", role=role)
    t = service.login(settings, "dana@example.com", temp, "127.0.0.1").token
    service.change_password(settings, t, temp, PW)
    return service.login(settings, "dana@example.com", PW, "127.0.0.1").token


def test_nothing_but_the_sign_in_screen_before_signing_in(settings):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [b.label for b in at.button] == ["Sign in"] and len(at.chat_input) == 0 and len(at.sidebar.button) == 0
    assert at.text_input[0].label == "Email" and at.text_input[1].label == "Password"


def test_a_wrong_password_shows_the_generic_message(settings):
    _signed_in(settings)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input("dana@example.com")
    at.text_input[1].input("not the password!!")
    at.button[0].click().run()
    assert at.error[0].value == service.GENERIC_ERROR and len(at.chat_input) == 0


def test_signed_in_user_sees_the_chat_and_a_stored_answer_renders(settings):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["token"] = _signed_in(settings)
    at.session_state["chat_history"] = [{"question": "How much does GQA save?", "answer": SAMPLE_ANSWER,
                                         "cost": 0.014, "latency_ms": 9000, "trace_id": "a" * 32}]
    at.run()
    assert not at.exception
    assert len(at.chat_input) == 1
    assert any("KV cache" in m.value for m in at.markdown)
    assert any("p. 19" in b.label for b in at.button)  # the citation chip
    assert any("800 KB" in m.value for m in at.markdown)  # the table block (rendered as Markdown)


def test_a_temporary_password_forces_the_change_screen(settings):
    temp = service.add_user(settings, "erin@example.com")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["token"] = service.login(settings, "erin@example.com", temp, "127.0.0.1").token
    at.run()
    assert at.title[0].value == "Change your password" and len(at.chat_input) == 0


def test_a_revoked_session_goes_back_to_sign_in(settings):
    token = _signed_in(settings)
    service.logout(settings, token)
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["token"] = token
    at.run()
    assert len(at.chat_input) == 0 and at.text_input[0].label == "Email"


def test_signing_in_through_the_form_opens_the_chat(settings):
    # the form used to be wiped on every rerun before it could be submitted
    _signed_in(settings)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input("Dana@Example.com")
    at.text_input[1].input(PW)
    at.button[0].click().run()
    assert not at.exception and len(at.chat_input) == 1 and "token" in at.session_state
