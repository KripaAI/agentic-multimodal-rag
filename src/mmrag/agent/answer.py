"""The answer contract (spec §5.4).

Two forms of every block:
- **model form** (`Answer`): what `compose` must return through structured output. Citations
  are ids only (`element_id` or `chunk_id`); the model never writes files, pages or boxes.
- **hydrated form** (`HydratedAnswer`): what the validator returns after looking every id up
  in PostgreSQL, adding `source_file`, `page` and `bbox` (and asset paths for images).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- model form (structured output)

class Citation(_Model):
    id: str  # an element_id (figure, table) or a chunk_id (text passage)


class TextBlock(_Model):
    type: Literal["text"] = "text"
    markdown: str
    citations: list[Citation]  # at least one, except in a "not found" answer (checked by the validator)


class ImageBlock(_Model):
    type: Literal["image"] = "image"
    element_id: str  # an original figure from the corpus (P3); its asset path comes from the DB
    short_caption: str
    citation: Citation


class ChartBlock(_Model):
    type: Literal["chart"] = "chart"
    chart_id: str  # returned by make_chart; the chart engine holds the validated spec


class TableBlock(_Model):
    type: Literal["table"] = "table"
    element_id: str  # a table from the corpus; rows come from doc_tables, not from the model
    citation: Citation


Block = Annotated[TextBlock | ImageBlock | ChartBlock | TableBlock, Field(discriminator="type")]


class Answer(_Model):
    """What `compose` returns. The sources list is built by the validator from the citations."""

    blocks: list[Block] = Field(min_length=1)
    not_found: bool = False  # true when the corpus does not answer the question (P5)
    missing: str | None = None  # what could not be found, at the round limit or when not_found


# ---------------------------------------------------------------- hydrated form (after validation)

class Location(_Model):
    element_id: str
    source_file: str
    page: int
    bbox: tuple[float, float, float, float]  # PDF points, origin top-left (spec §7.5)


class HydratedCitation(_Model):
    id: str
    locations: list[Location]  # one for an element_id; one per covered element for a chunk_id


class HydratedBlock(_Model):
    type: Literal["text", "image", "chart", "table"]
    content: dict  # markdown · {asset_path, short_caption} · chart spec + data table · {columns, rows}
    citations: list[HydratedCitation]
    approximate: bool = False  # charts with any estimated value (P4)


class Source(_Model):
    source_file: str
    page: int
    element_id: str


class HydratedAnswer(_Model):
    blocks: list[HydratedBlock]
    sources: list[Source]
    notices: list[str] = []  # e.g. "a chart was removed because a value could not be verified"
