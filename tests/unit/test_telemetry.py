"""JSON logs carry trace ids; telemetry never breaks the app (LLD §5.9, NFR-12)."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
import time

import pytest
from opentelemetry.sdk.trace import TracerProvider

from mmrag.obs.telemetry import JsonFormatter

pytestmark = pytest.mark.unit


def _format(msg: str = "hello", exc_info=None) -> dict:
    record = logging.LogRecord("mmrag.test", logging.INFO, __file__, 1, msg, None, exc_info)
    return json.loads(JsonFormatter().format(record))


def test_log_line_outside_span_has_no_trace_id():
    entry = _format()
    assert entry["msg"] == "hello"
    assert entry["level"] == "INFO"
    assert entry["trace_id"] is None and entry["span_id"] is None


def test_log_line_inside_span_carries_its_trace_and_span_id():
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("unit") as span:
        ctx = span.get_span_context()
        entry = _format()
    assert entry["trace_id"] == format(ctx.trace_id, "032x")
    assert entry["span_id"] == format(ctx.span_id, "016x")


def test_log_line_includes_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        entry = _format("failed", sys.exc_info())
    assert "ValueError: boom" in entry["exc"]


def test_unreachable_trace_viewer_does_not_break_the_app(base_config, write_config):
    """Runs in a subprocess: OpenTelemetry providers are process-global."""
    base_config["observability"].update(
        otlp_traces_endpoint="http://127.0.0.1:9/v1/traces",  # nothing listens here
        log_file=None,
        capture_content=False,
    )
    cfg = write_config(base_config)
    script = textwrap.dedent(f"""
        from pathlib import Path
        from mmrag.config import load_settings
        from mmrag.obs import get_logger, get_tracer, init_telemetry, shutdown_telemetry
        s = load_settings(Path(r"{cfg}"), {{"DATABASE_URL": "postgresql://u:p@localhost:1/x"}})
        init_telemetry(s, "test")
        with get_tracer("t").start_as_current_span("work"):
            get_logger("t").info("still working")
        shutdown_telemetry()
        print("APP-OK")
    """)
    started = time.monotonic()
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "APP-OK" in r.stdout
    assert time.monotonic() - started < 30
