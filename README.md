# mmrag — Agentic Multimodal RAG

Answers questions over the PDFs in `data/pdfs/` with text, original figures, charts and tables, each cited to its source page. Design documents are in [docs/](docs/):

- [constitution](docs/01-constitution.md)
- [technical specification](docs/02-technical-specification.md)
- [implementation plan](docs/03-implementation-plan.md)
- [low-level design](docs/05-low-level-design.md)
- [evaluation results](docs/06-evaluation-results.md)
- [operations guide](docs/07-operations.md): backups, managing PDFs, accounts, alerts, HTTPS deployment

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

   Then set `agent.model`, `agent.summary_model` and `eval.judge_model` in `config.yaml`. The defaults are `gpt-5.4-mini` for the agent and summaries, and `gpt-4.1-mini` for the evaluation judge. Prices go under `pricing`, so costs are reported.

## From a PDF to answers

Each PDF goes through four steps. Each step processes that PDF only:

```
mmrag doc add path\to\file.pdf          # 1. copy into data/pdfs/ and parse (prints a review sheet)
mmrag caption bundle file.pdf           # 2. figure descriptions on a free Kaggle GPU
mmrag caption push file.pdf             #    (needs KAGGLE_API_TOKEN in .env)
mmrag caption status file.pdf           #    wait until it says COMPLETE
mmrag caption pull file.pdf
mmrag caption import file.pdf
mmrag ingest index file.pdf             # 3. tables summarized, embedded, written to PostgreSQL
mmrag ask "What does the RAG pipeline diagram show?"   # 4. ask (or use the chat app below)
```

A PDF without figures can skip step 2. `mmrag doc list | replace | remove` manages the library; see the [operations guide](docs/07-operations.md).

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

## The chat app (Phase 7)

```powershell
python scripts/fetch_common_passwords.py      # once: the password blocklist (kept local)
mmrag db migrate                              # users, sessions, auth_events
mmrag user add you@example.com --role admin   # prints a temporary password once
streamlit run app/ui.py                       # http://127.0.0.1:8501 (localhost only)
```

At the first sign-in you choose your own password (at least 12 characters, not a common one).

Accounts are managed from the command line only:

```powershell
mmrag user add | reset-password | disable | enable | unlock <email>
mmrag user list
```

Daily limits per user (questions and US$) are set in `config.yaml`, under `auth`. For production, serve the app over HTTPS behind a reverse proxy.

## Tests

Development is test-driven where it fits (constitution W5, LLD §10). Install the test tools with `pip install -r requirements-dev.txt`.

| Command | Runs |
|---|---|
| `.venv\Scripts\python -m pytest` | Unit and integration tests. Needs `docker compose up -d`; each integration test uses its own throwaway database. |
| `.venv\Scripts\python -m pytest -m unit` | Fast tests only, no services needed |
| `.venv\Scripts\python -m pytest -m regression` | Re-parses the approved PDFs and compares with `tests/regression/snapshots/` (skipped if the PDFs are absent). After approving a detection change on the review sheet, refresh with `python -m tests.regression.snapshot` |
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
