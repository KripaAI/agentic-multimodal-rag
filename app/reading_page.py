"""Birth-chart reading page (Phase 11). The chart is computed by code; the fine-tuned model
(astro-v2-8b on Modal) writes the reading from the chart facts and library passages."""

from __future__ import annotations

from datetime import date, time

import streamlit as st

from mmrag.auth import service
from mmrag.config import Settings


def _table(header: list[str], rows: list[list]) -> None:
    """A Markdown table (st.dataframe needs pandas, which Application Control blocks here)."""
    esc = lambda v: str(v if v is not None else "").replace("|", "\\|")  # noqa: E731
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    st.markdown("\n".join(lines + ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows]))


def _chart_table(chart) -> None:
    from mmrag.astro.natal import BODIES, dms

    rows = [["Lagna", chart.lagna.sign, dms(chart.lagna.degree), 1, f"{chart.lagna.nakshatra} {chart.lagna.pada}", ""]]
    for b in BODIES:
        p = chart.placements[b]
        notes = ", ".join(n for n in ("retrograde" if p.retrograde and b not in ("Rahu", "Ketu") else "",
                                      p.dignity or "", "combust" if p.combust else "") if n)
        rows.append([b, p.sign, dms(p.degree), p.house, f"{p.nakshatra} {p.pada}", notes])
    _table(["", "sign", "degree", "house", "nakshatra pada", "notes"], rows)


def _wake_writer(settings: Settings) -> None:
    """Start the GPU while the user types the birth details: any request makes Modal start the
    server, which takes a minute or two from cold. At most once every 4 minutes (it stops after 5 idle)."""
    import threading
    import time as _time
    import urllib.request

    base = (settings.secrets.writer_base_url or "").rstrip("/")
    if not base or _time.monotonic() - st.session_state.get("writer_woken", -1e9) < 240:
        return
    st.session_state["writer_woken"] = _time.monotonic()
    url = base[:-3] + "/health" if base.endswith("/v1") else base + "/health"

    def ping() -> None:
        try:
            urllib.request.urlopen(url, timeout=600).read()
        except Exception:  # noqa: BLE001 - only a warm-up
            pass

    threading.Thread(target=ping, daemon=True, name="writer-wake-up").start()


def reading_view(settings: Settings, user: service.User) -> None:
    from mmrag.astro.natal import ChartInputError, compute_chart, parse_birth
    from mmrag.astro.reading import WriterNotConfigured, sources_markdown, write_reading

    st.title("Birth chart reading")
    if st.button("← Back to chat"):
        st.session_state.pop("view", None)
        st.rerun()
    st.caption("The chart is calculated by software (sidereal, Lahiri ayanamsa, whole-sign houses). The reading "
               "is written by the fine-tuned astrology model from those facts and passages from the library. "
               "Astrology describes tendencies, not certainties; it is no substitute for medical, legal or "
               "financial advice.")
    astro_ready = bool(settings.secrets.writer_base_url and settings.secrets.writer_api_key)
    if astro_ready:
        _wake_writer(settings)
    with st.form("birth"):
        c = st.columns(2)
        name = c[0].text_input("Name (optional)")
        place = c[1].text_input("Place of birth (shown only)")
        c = st.columns(3)
        born = c[0].date_input("Date of birth", value=date(1990, 1, 1), min_value=date(1850, 1, 1),
                               max_value=date(2149, 12, 31))
        at = c[1].time_input("Time of birth (local)", value=time(12, 0), step=60)
        tz = c[2].text_input("Time zone", value="Asia/Kolkata", help="A zone name such as Asia/Kolkata, "
                             "or an offset such as +05:30")
        c = st.columns(2)
        lat = c[0].number_input("Latitude (north +)", min_value=-66.0, max_value=66.0, value=28.6139, format="%.4f")
        lon = c[1].number_input("Longitude (east +)", min_value=-180.0, max_value=180.0, value=77.2090, format="%.4f")
        question = st.text_input("Question about this chart (optional; empty for a full reading)")
        writers = ([f"{settings.writer.model} (fine-tuned)"] if astro_ready else []) + [f"{settings.agent.model} (to compare)"]
        writer = st.radio("Writer", writers, horizontal=True)
        use_books = st.checkbox("Use passages from the library", value=True)
        submitted = st.form_submit_button("Read the chart", type="primary")
    if not astro_ready:
        st.info("The fine-tuned model is not configured: set WRITER_BASE_URL and WRITER_API_KEY in .env.")
    if submitted:
        try:
            service.check_limits(settings, user.user_id)
        except service.LimitReached as e:
            st.warning(str(e))
            return
        try:
            chart = compute_chart(parse_birth(born.isoformat(), at.strftime("%H:%M"), tz, lat, lon, name, place))
        except ChartInputError as e:
            st.error(str(e))
            return
        with st.status("Finding passages and writing the reading… (the first one after a pause also starts "
                       "the GPU, up to a few minutes)", expanded=False) as status:
            try:
                reading = write_reading(settings, chart, question=question or None,
                                        writer="astro" if writer.startswith("astro") else "openai",
                                        use_books=use_books)
            except WriterNotConfigured as e:
                status.update(label="Not configured", state="error")
                st.error(str(e))
                return
            except Exception as e:  # noqa: BLE001 - keep the app alive
                status.update(label="Something went wrong", state="error")
                st.error(f"The reading could not be written ({type(e).__name__}: {e}).")
                return
            status.update(label=f"Done in {reading.seconds:.0f} s", state="complete")
        st.session_state["reading"] = reading
    reading = st.session_state.get("reading")
    if reading is None:
        return
    st.subheader("The chart")
    _chart_table(reading.chart)
    with st.expander("All chart facts given to the writer"):
        st.text(reading.facts)
    st.subheader("The reading")
    for n in reading.notices:
        st.warning(n)
    st.markdown(reading.text or "_(no text)_")
    if reading.cited:
        with st.expander(f"Sources ({len(reading.cited)} passages)"):
            st.markdown(sources_markdown(reading).replace("\n", "  \n"))
            for p in reading.cited:
                st.caption(f"[{p.n}] {p.text}")
    st.caption(f"{reading.model} · {reading.seconds:.0f} s · {reading.input_tokens:,} tokens in, "
               f"{reading.output_tokens:,} out · {len(reading.passages)} passages given, {len(reading.cited)} cited")
