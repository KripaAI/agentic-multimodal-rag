"""The chat app (plan Phase 7, LLD §5.7). Run with:  streamlit run app/ui.py

Streamlit re-runs this script on every click. Only a new question runs the agent; everything
else (opening a source, a table, an earlier conversation) renders from st.session_state or from
query_log, never re-running the agent, validator or search.
"""

from __future__ import annotations

import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # app/login.py
import login  # noqa: E402

from mmrag.auth import service  # noqa: E402
from mmrag.config import Settings  # noqa: E402

st.set_page_config(page_title="Document assistant", page_icon="📚", layout="wide")


@st.cache_resource
def _settings_for(config_file: str, database_url: str) -> Settings:
    """Built from exactly this config file and database (get_settings() is process-cached)."""
    import os

    from mmrag.config import PROJECT_ROOT, load_settings
    from mmrag.obs import init_telemetry

    s = load_settings(Path(config_file) if config_file else PROJECT_ROOT / "config.yaml", os.environ)
    init_telemetry(s, service_name="mmrag-ui")  # once per server process and configuration, not per rerun
    return s


def _settings() -> Settings:
    import os

    from dotenv import load_dotenv

    from mmrag.config import PROJECT_ROOT

    load_dotenv(PROJECT_ROOT / ".env")
    return _settings_for(os.environ.get("MMRAG_CONFIG", ""), os.environ.get("DATABASE_URL", ""))


@st.cache_data(max_entries=64, show_spinner=False)
def _page_png(pdf: str, page: int, bboxes: tuple, dpi: int) -> bytes:
    from mmrag.ui.pages import page_with_highlights

    return page_with_highlights(Path(pdf), page, [list(b) for b in bboxes], dpi)


# ---------------------------------------------------------------- answer rendering

def _md_table(header: list, rows: list[list]) -> None:
    """A Markdown table. Not st.dataframe/st.table: those load pandas, whose DLLs Windows
    Application Control blocks on this machine (Phase 7)."""
    def esc(v) -> str:
        return str(v if v is not None else "").replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(esc(h) or " " for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(esc(v) for v in row) + " |" for row in rows]
    st.markdown("\n".join(lines))


def _chips(citations: list[dict], key: str) -> None:
    """One button per cited page; clicking opens it in the source panel (no agent re-run)."""
    pages: dict[tuple[str, int], list] = defaultdict(list)
    for c in citations:
        for loc in c.get("locations", []):
            pages[(loc["source_file"], loc["page"])].append(tuple(loc["bbox"]))
    if not pages:
        return
    cols = st.columns(min(len(pages), 4))
    for n, ((file, page), boxes) in enumerate(pages.items()):
        label = f"📄 {Path(file).stem[:22]} · p. {page}"
        if cols[n % len(cols)].button(label, key=f"{key}-chip-{n}", help=f"{file}, page {page}"):
            st.session_state["source"] = {"file": file, "page": page, "bboxes": boxes}
            st.rerun()


def render_answer(answer: dict, key: str, settings: Settings) -> None:
    data_dir = settings.resolve(settings.paths.data_dir)
    for notice in answer.get("notices", []):
        st.warning(notice)
    for n, b in enumerate(answer.get("blocks", [])):
        c, k = b["content"], f"{key}-b{n}"
        if b["type"] == "text":
            st.markdown(c["markdown"])
        elif b["type"] == "image":
            path = data_dir / c["asset_path"] if c.get("asset_path") else None
            if path and path.is_file():
                st.image(str(path), caption=c.get("short_caption"))  # full-screen button to enlarge
            else:
                st.caption(f"Figure unavailable: {c.get('short_caption', '')}")
        elif b["type"] == "chart":
            import plotly.io as pio

            if b.get("approximate"):
                st.caption("⚠️ Approximate: some values are estimated from a figure.")
            st.plotly_chart(pio.from_json(json.dumps(c["spec"])), key=f"{k}-chart", width="stretch")
            for note in c.get("notes") or []:
                st.caption(note)
            table = c.get("data_table") or []
            if table:
                with st.expander("Data table"):
                    _md_table(table[0], table[1:])
        elif b["type"] == "table":
            if c.get("title"):
                st.markdown(f"**{c['title']}**")
            cols = c.get("columns") or [f"col {i + 1}" for i in range(len(c["rows"][0]) if c.get("rows") else 0)]
            _md_table(cols, c.get("rows", []))
        _chips(b.get("citations", []), k)
    if answer.get("sources"):
        with st.expander(f"Sources ({len(answer['sources'])})"):
            for s in answer["sources"]:
                st.markdown(f"- {s['source_file']}, p. {s['page']}")


def source_panel(settings: Settings) -> None:
    src = st.session_state["source"]
    top = st.columns([4, 1])
    top[0].markdown(f"**{src['file']}**, page {src['page']}")
    if top[1].button("✕", key="close-source", help="Close"):
        st.session_state.pop("source")
        st.rerun()
    pdf = settings.resolve(settings.paths.pdf_dir) / src["file"]
    if not pdf.is_file():
        st.error("The PDF is not available on this server.")
        return
    st.image(_page_png(str(pdf), src["page"], tuple(tuple(b) for b in src["bboxes"]), settings.ui.page_dpi),
             width="stretch")


# ---------------------------------------------------------------- chat

def ask(settings: Settings, user: service.User, question: str) -> None:
    from mmrag.agent.graph import run_query

    try:
        service.check_limits(settings, user.user_id)
    except service.LimitReached as e:
        st.warning(str(e))  # no API call is made
        return
    thread = st.session_state.setdefault("thread_id", uuid.uuid4().hex[:12])
    with st.status("Working…", expanded=False) as status:
        try:
            run = run_query(question, settings, thread_id=thread, user_id=user.user_id,
                            on_step=lambda label: status.update(label=f"{label}…"))
        except Exception as e:  # noqa: BLE001 - show a clear message, keep the app alive
            status.update(label="Something went wrong", state="error")
            st.error(f"The question could not be answered ({type(e).__name__}). Please try again.")
            return
        status.update(label=f"Done in {run.latency_ms / 1000:.1f} s", state="complete")
    st.session_state.setdefault("chat_history", []).append({
        "question": question, "answer": run.answer.model_dump(mode="json"), "cost": run.cost_usd,
        "latency_ms": run.latency_ms, "trace_id": run.trace_id})
    st.session_state["session_cost"] = st.session_state.get("session_cost", 0.0) + (run.cost_usd or 0.0)
    st.session_state["session_tokens"] = (st.session_state.get("session_tokens", 0)
                                          + run.input_tokens + run.output_tokens)
    st.rerun()


def sidebar(settings: Settings, user: service.User) -> None:
    from mmrag.ui.data import thread_turns, user_threads

    with st.sidebar:
        st.markdown(f"**{user.email}**" + (" · admin" if user.is_admin else ""))
        if st.button("➕ New conversation", width="stretch"):
            for k in ("thread_id", "chat_history", "source"):
                st.session_state.pop(k, None)
            st.session_state.pop("view", None)
            st.rerun()
        st.subheader("Conversations")
        for t in user_threads(settings, user.user_id, limit=20):
            label = (t.title[:38] + "…") if len(t.title) > 38 else t.title
            if st.button(label, key=f"thread-{t.thread_id}", width="stretch",
                         type="primary" if t.thread_id == st.session_state.get("thread_id") else "secondary"):
                st.session_state["thread_id"] = t.thread_id
                st.session_state["chat_history"] = [
                    {"question": x.question, "answer": x.answer, "cost": x.cost_usd, "latency_ms": x.latency_ms,
                     "trace_id": x.trace_id} for x in thread_turns(settings, user.user_id, t.thread_id)]
                st.session_state.pop("source", None)
                st.session_state.pop("view", None)
                st.rerun()
        st.divider()
        n, cost, _ = service.usage_today(settings, user.user_id)
        a = settings.auth
        st.caption(f"Today: {n}/{a.daily_question_limit} questions · ${cost:.3f}/${a.daily_cost_limit_usd:.2f}")
        st.caption(f"This session: ${st.session_state.get('session_cost', 0.0):.4f} · "
                   f"{st.session_state.get('session_tokens', 0):,} tokens")
        if user.is_admin and st.button("📊 Admin metrics", width="stretch"):
            st.session_state["view"] = "admin"
            st.rerun()
        if st.button("What I remember", width="stretch"):
            st.session_state["view"] = "memory"
            st.rerun()
        if st.button("Change password", width="stretch"):
            st.session_state["view"] = "password"
            st.rerun()
        if st.button("Sign out", width="stretch"):
            service.logout(settings, st.session_state.get("token"))
            st.session_state.clear()
            st.rerun()


def memory_view(settings: Settings, user: service.User) -> None:
    """What the assistant remembers about this user, and the controls over it (FR-25).

    These notes are never evidence: they shape how a question is understood and how an answer
    is presented, and the validator refuses any answer that tries to cite one (P13).
    """
    from mmrag import memory

    st.title("What the assistant remembers")
    if st.button("Back to chat"):
        st.session_state.pop("view", None)
        st.rerun()
    if not settings.memory.enabled:
        st.info("Memory is switched off for the whole application.")
        return
    enabled = memory.is_enabled(settings, user.user_id)
    st.caption("Notes about you from earlier conversations: what you work on and how you like answers. "
               "They are never used as a source: every fact in an answer still comes from the documents.")
    if st.toggle("Remember me between conversations", value=enabled, key="memory-toggle") != enabled:
        memory.set_enabled(settings, user.user_id, not enabled)
        st.rerun()
    if not enabled:
        st.warning("Memory is off: nothing new is stored and nothing is recalled. What was stored before is kept "
                   "until you delete it.")
    with memory.open_store(settings) as store:
        items = memory.list_memories(store, user.user_id)
        if not items:
            st.info("Nothing is remembered about you yet.")
            return
        for m in items:
            row = st.columns([8, 1])
            label = "About you" if m.kind == "semantic" else "A past conversation"
            row[0].markdown(f"**{label}** · {(m.updated_at or '')[:10]}  \n{m.text}")
            if row[1].button("Delete", key=f"mem-{m.kind}-{m.key}"):
                memory.delete(store, user.user_id, m.kind, m.key)
                st.rerun()
        st.divider()
        if st.button(f"Delete all {len(items)} memories", type="primary"):
            memory.forget_all(store, user.user_id)
            st.rerun()


def admin_view(settings: Settings) -> None:
    from mmrag.ui.data import admin_metrics

    st.title("Admin metrics (last 14 days)")
    if st.button("← Back to chat"):
        st.session_state.pop("view", None)
        st.rerun()
    m = admin_metrics(settings)
    endpoint = settings.observability.otlp_traces_endpoint or ""
    phoenix = endpoint.rsplit("/v1/", 1)[0] if endpoint else ""
    c = st.columns(3)
    c[0].metric("Questions", sum(n for _, n in m["questions_per_day"]))
    c[1].metric("Answers with removed parts", "–" if m["answers_with_removed_parts"] is None
                else f"{m['answers_with_removed_parts']:.0%}")
    c[2].metric("Failed sign-ins", sum(n for _, n in m["failed_sign_ins_per_day"]))
    st.subheader("Questions per day")
    _md_table(["day", "questions"], [[d, n] for d, n in m["questions_per_day"]])
    st.subheader("Slowest answers")
    _md_table(["question", "seconds", "when", "trace"],
              [[q[:80], round(ms / 1000, 1), str(t)[:16], f"[open]({phoenix}/redirects/traces/{tr})" if phoenix else tr]
               for q, ms, tr, t in m["slowest"]])
    st.subheader("Cost per user")
    _md_table(["user", "questions", "cost (US$)"], [[e, n, round(cst, 4)] for e, n, cst in m["cost_per_user"]])
    st.subheader("Failed sign-ins per day")
    _md_table(["day", "failures"], [[d, n] for d, n in m["failed_sign_ins_per_day"]])


def chat_view(settings: Settings, user: service.User) -> None:
    has_source = "source" in st.session_state
    main, side = st.columns([3, 2]) if has_source else (st.container(), None)
    with main:
        history = st.session_state.get("chat_history", [])
        if not history:
            st.title("Ask about the documents")
            st.caption("Answers cite their sources. Click a 📄 chip to see the page with the passage highlighted.")
        for n, turn in enumerate(history):
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                render_answer(turn["answer"], f"t{n}", settings)
                meta = f"{turn['latency_ms'] / 1000:.1f} s"
                if turn.get("cost") is not None:
                    meta += f" · ${turn['cost']:.4f}"
                st.caption(meta)
    if side is not None:
        with side:
            source_panel(settings)
    question = st.chat_input("Ask a question about the documents")
    if question:
        with main:
            with st.chat_message("user"):
                st.markdown(question)
            ask(settings, user, question)


def main() -> None:
    settings = _settings()
    user = service.require_user(settings, st.session_state.get("token"))  # every rerun
    if user is None:
        if "token" in st.session_state:  # expired or revoked: drop everything from that session
            note = st.session_state.get("_signed_out_message")
            st.session_state.clear()
            if note:
                st.session_state["_signed_out_message"] = note
        login.sign_in_screen(settings)  # (never clear on every rerun: that would wipe the form being submitted)
        return
    if user.must_change_password:
        login.change_password_screen(settings, forced=True)
        return
    sidebar(settings, user)
    view = st.session_state.get("view")
    if view == "password":
        login.change_password_screen(settings, forced=False)
    elif view == "memory":
        memory_view(settings, user)
    elif view == "admin" and user.is_admin:
        admin_view(settings)
    else:
        chat_view(settings, user)


main()
