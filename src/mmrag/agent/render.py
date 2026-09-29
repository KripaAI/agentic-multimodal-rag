"""Answer page: one self-contained HTML file per question (plan Phase 4, task 6).

Renders the hydrated blocks in order: text with citation chips, original figures with
captions, interactive Plotly charts with their data tables and "approximate" label, tables,
then the sources list. Each citation shows file and page. Written to data/answers/.
"""

from __future__ import annotations

from pathlib import Path

from mmrag.agent.graph import QueryRun
from mmrag.config import Settings


def write_answer_page(question: str, run: QueryRun, settings: Settings) -> Path:
    """Write data/answers/<timestamp>-<slug>.html and return its path. The footer shows the
    thread, rounds, tokens, cost, latency and the Phoenix trace link."""
    raise NotImplementedError
