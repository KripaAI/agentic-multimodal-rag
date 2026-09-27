"""OpenTelemetry setup: traces, metrics and JSON logs with trace ids (LLD §5.9).

Telemetry must never break the application: export is batched and
non-blocking, and setup failures degrade to a no-op with a warning.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

import mmrag
from mmrag.config import Settings

_initialized = False
_log = logging.getLogger("mmrag.obs")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, carrying the active trace and span ids."""

    def format(self, record: logging.LogRecord) -> str:
        ctx = trace.get_current_span().get_span_context()
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": format(ctx.trace_id, "032x") if ctx.is_valid else None,
            "span_id": format(ctx.span_id, "016x") if ctx.is_valid else None,
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def _setup_logging(settings: Settings) -> None:
    obs = settings.observability
    root = logging.getLogger()
    root.setLevel(obs.log_level)
    for h in list(root.handlers):
        root.removeHandler(h)
    formatter = JsonFormatter()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if obs.log_file:
        path = settings.resolve(obs.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.TimedRotatingFileHandler(
            path, when="midnight", backupCount=30, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)

    # Exporter errors are expected when the trace viewer is down; keep them quiet.
    logging.getLogger("opentelemetry").setLevel(logging.ERROR)


def _setup_tracing(settings: Settings, resource: Resource) -> None:
    obs = settings.observability
    provider = TracerProvider(
        resource=resource, sampler=ParentBased(TraceIdRatioBased(obs.sample_ratio))
    )
    if obs.otlp_traces_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=obs.otlp_traces_endpoint, timeout=5))
        )
    trace.set_tracer_provider(provider)


def _setup_metrics(settings: Settings, resource: Resource) -> None:
    from opentelemetry.sdk.metrics import MeterProvider

    readers = []
    endpoint = settings.observability.otlp_metrics_endpoint
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint)))
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))


def _instrument_libraries(settings: Settings) -> None:
    """Auto-instrument psycopg (SQL spans) and the OpenAI SDK (GenAI spans)."""
    try:
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

        PsycopgInstrumentor().instrument(enable_commenter=False)
    except Exception as e:  # never let telemetry break the app
        _log.warning("psycopg instrumentation unavailable: %s", e)
    try:
        import os

        if settings.observability.capture_content:
            os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
    except Exception as e:
        _log.warning("OpenAI instrumentation unavailable: %s", e)


def init_telemetry(settings: Settings, service_name: str) -> None:
    """Initialize once per process. Safe to call repeatedly."""
    global _initialized
    if _initialized:
        return
    _setup_logging(settings)
    if settings.observability.enabled:
        try:
            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.version": mmrag.__version__,
                    "deployment.environment": settings.observability.environment,
                }
            )
            _setup_tracing(settings, resource)
            _setup_metrics(settings, resource)
            _instrument_libraries(settings)
        except Exception as e:
            _log.warning("Telemetry disabled, setup failed: %s", e)
    _initialized = True


def shutdown_telemetry() -> None:
    """Flush pending spans and metrics (call before a CLI process exits)."""
    for provider in (trace.get_tracer_provider(), metrics.get_meter_provider()):
        shutdown = getattr(provider, "shutdown", None)
        if shutdown:
            try:
                shutdown()
            except Exception as e:
                _log.warning("Telemetry shutdown failed: %s", e)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name, mmrag.__version__)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
