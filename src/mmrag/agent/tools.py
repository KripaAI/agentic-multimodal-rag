"""Agent tools (spec §7.1, LLD §5.3): plain Python functions the model may call.

`TOOL_SPECS` describes them to OpenAI (function calling). `dispatch` runs the calls of one
round in parallel: it validates the arguments, runs each function with a timeout, records
results in the evidence ledger, and turns any error into an error result for the model
instead of raising. No MCP server (D16).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mmrag.agent.ledger import EvidenceLedger
from mmrag.config import Settings

TOOL_SPECS: list[dict] = []  # OpenAI tool definitions (name, description, JSON schema), one per tool


@dataclass
class ToolContext:
    """What every tool may use during one question."""

    settings: Settings
    ledger: EvidenceLedger
    embed_query: object  # callable(str) -> vector, cached per query text
    chart_dir: Path  # where make_chart writes PNGs for this question


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str  # JSON text sent back to the model
    is_error: bool = False


# ---------------------------------------------------------------- search (Phase 3 hybrid search)

def search_text(ctx: ToolContext, query: str, k: int = 8) -> list[dict]:
    """Top text chunks: chunk_id, text, section_path, source_file, page, related items."""
    raise NotImplementedError


def search_figures(ctx: ToolContext, query: str, k: int = 8) -> list[dict]:
    """Top figures: element_id, short caption, description excerpt, page, related paragraphs."""
    raise NotImplementedError


def search_tables(ctx: ToolContext, query: str, k: int = 8) -> list[dict]:
    """Top tables: element_id, title, summary, columns, page, related paragraphs."""
    raise NotImplementedError


# ---------------------------------------------------------------- fetch

def get_figure(ctx: ToolContext, element_id: str) -> dict:
    """The full caption record (description, visible text, chart values with exact/estimated
    flags); the figure image is attached for the model to view when useful."""
    raise NotImplementedError


def get_table(ctx: ToolContext, element_id: str) -> dict:
    """The full structured table from doc_tables: columns, every row, units, title."""
    raise NotImplementedError


def view_page(ctx: ToolContext, source_file: str, page: int) -> dict:
    """A rendered page image (at ui.page_dpi) for the model to look at. Only files that are
    in `documents` are accepted; resolved paths must stay under data/ (LLD §9)."""
    raise NotImplementedError


# ---------------------------------------------------------------- calculate and chart

def compute_tool(ctx: ToolContext, expression: str, refs: dict[str, str]) -> dict:
    """Safe arithmetic (agent/compute.py). `refs` maps each name to an evidence id; the value is
    looked up in the ledger. The result is recorded in the ledger so charts may use it."""
    raise NotImplementedError


def make_chart_tool(ctx: ToolContext, **request) -> dict:
    """charts.engine.make_chart; returns a chart_id on success, or the reasons it was rejected."""
    raise NotImplementedError


TOOLS = {
    "search_text": search_text,
    "search_figures": search_figures,
    "search_tables": search_tables,
    "get_figure": get_figure,
    "get_table": get_table,
    "view_page": view_page,
    "compute": compute_tool,
    "make_chart": make_chart_tool,
}


def dispatch(calls: list[ToolCall], ctx: ToolContext) -> list[ToolResult]:
    """Run one round's calls in parallel (thread pool, `agent.tool_timeout_s` each), in the
    order given. Invalid arguments, unknown tools, exceptions and timeouts become error results."""
    raise NotImplementedError
