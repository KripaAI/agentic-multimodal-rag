# mmrag — Agentic Multimodal RAG

Answers questions over the PDFs in `data/pdfs/` with text, original figures, charts and tables, each cited to its source page. Design documents are in [docs/](docs/):

- [constitution](docs/01-constitution.md)
- [technical specification](docs/02-technical-specification.md)
- [implementation plan](docs/03-implementation-plan.md)
- [low-level design](docs/05-low-level-design.md)

## Setup (Windows, Phase 0)

Requirements: Python 3.11, Docker Desktop.

1. Create the environment:

   ```
   py -3.11 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   .venv\Scripts\python -m pip install -e .
   ```

2. Configure secrets:
   - Copy `.env.example` to `.env`.
   - Set a Postgres password; the same password goes in `DATABASE_URL`.
   - Add `OPENAI_API_KEY`: an API key from platform.openai.com, not a ChatGPT subscription.
3. Start PostgreSQL + pgvector (port 5433) and the Phoenix trace viewer (http://localhost:6006):

   ```
   docker compose up -d
   ```

4. Apply database migrations and verify everything:

   ```
   .venv\Scripts\mmrag db migrate
   .venv\Scripts\mmrag check
   ```

   `check` verifies config, the database (pgvector + HNSW) and OpenAI access, and sends a test trace to Phoenix.
5. Pick the agent and judge models from the ones your key can use:

   ```
   .venv\Scripts\mmrag models
   ```

   Then set `agent.model`, `agent.summary_model` and `eval.judge_model` in `config.yaml`.

## Parsing (Phase 1)

```
.venv\Scripts\mmrag profile                                  # pages, images, tables, textless pages per PDF
.venv\Scripts\mmrag ingest parse Buildig-multimodal-rag.pdf  # parse one PDF from data/pdfs/
```

`ingest parse` writes, per document (`{doc_id}` = first 16 hex of the PDF's SHA-256):

| File | Contents |
|---|---|
| `data/elements/{doc_id}/elements.jsonl` | Every text block, image, vector figure, table and scanned page, with page, bbox and ID |
| `data/elements/{doc_id}/skip_log.jsonl` | Everything dropped on purpose (headers/footers, repeated images), with the reason |
| `data/elements/{doc_id}/rejected_regions.jsonl` | Drawing clusters the figure filters rejected |
| `data/elements/{doc_id}/review_sheet.html` | Open in a browser to check detection page by page |
| `data/tables/{doc_id}/tables.jsonl` | Tables as columns and rows |
| `data/assets/{doc_id}/*.png` | Figure, image and scanned-page PNGs |

Detection thresholds are in `config.yaml` under `parse:`.

## Tests

Development is test-driven where it fits (constitution W5, LLD §10). Install the test tools with `pip install -r requirements-dev.txt`.

| Command | Runs |
|---|---|
| `.venv\Scripts\python -m pytest` | Unit and integration tests. Needs `docker compose up -d`; each integration test uses its own throwaway database. |
| `.venv\Scripts\python -m pytest -m unit` | Fast tests only, no services needed |
| `.venv\Scripts\python -m pytest -m live` | Real OpenAI calls (costs a fraction of a cent); run on request |
| `.venv\Scripts\python -m pytest --cov=mmrag` | With a coverage report |

## Layout

| Path | Contents |
|---|---|
| `config.yaml` | All tunable settings (no secrets) |
| `src/mmrag/` | Application code |
| `db/migrations/` | Versioned SQL schema files |
| `tests/` | `unit/`, `integration/`, `live/` test suites; `fixtures/` for sample pages (Phase 1) |
| `data/pdfs/` | Source PDFs (the knowledge base) |
| `data/` (other folders) | Generated files: elements, figures, captions, caches, logs |
| `gpu_job/` | Kaggle/Colab captioning notebook (Phase 2) |
| `eval/` | Golden set and evaluation (Phase 6) |
| `app/` | Streamlit UI (Phase 7) |

Database files live in the Docker volume `pgdata`. Back up with `pg_dump`, not by copying files.
