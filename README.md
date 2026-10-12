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

If Windows Smart App Control blocks a library ("An Application Control policy has blocked this file"), run the ingestion command in Linux instead. Build once with `docker compose --profile tools build ingest`, then use for example `docker compose run --rm ingest ingest index "file.pdf"`.

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
mmrag user delete <email> --yes               # account, sessions and memories
```

Daily limits per user (questions and US$) are set in `config.yaml`, under `auth`. For production, serve the app over HTTPS behind a reverse proxy.

## What the assistant remembers (Phase 9)

Each user gets a few durable notes — what they work on, how they like answers — and a short summary of each finished conversation, so a preference stated once carries over and "the chart you showed me last week" means something. Memory **personalises; it never informs**: no answer may cite a memory, and the validator refuses any that tries. Every fact still comes from the documents, with a citation.

In the app, **What I remember** in the sidebar lists everything kept about you, deletes any of it, and switches memory off. From the command line:

```powershell
mmrag memory list <email>                     # everything remembered, with its keys
mmrag memory add output_format "Prefers charts to tables." <email>
mmrag memory delete semantic output_format <email>
mmrag memory forget-all <email> --yes
mmrag memory off | on <email>                 # off stops storing and recalling
mmrag memory summarize <email>                # summarise finished conversations (run nightly)
```

Settings are under `memory` in `config.yaml`; `memory.enabled: false` switches it off for everyone. Operations are in [docs/07-operations.md](docs/07-operations.md) §9.

## Trying another embedding model (Phase 10)

Documents and questions must be embedded by the same model, and vectors from two models cannot be
compared. So an experiment gets **its own database** on the same server, and its own config file
that holds only what differs from `config.yaml`. The parsed PDFs and figure captions are reused:
only `ingest index` runs again. The production database is never touched.

`config.gemma.yaml` tries [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), an
open model that runs on the CPU (text only, 768 dims), in a database called `mmrag_gemma`:

```powershell
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements-local-embed.txt
$env:MMRAG_CONFIG = "config.gemma.yaml"       # this shell only; every mmrag command now uses it
mmrag db create                               # creates mmrag_gemma
mmrag db migrate                              # builds vector(768) columns from embed.dims
mmrag check                                   # loads the model (downloaded once, ~1 GB) and checks dims
mmrag ingest index file.pdf                   # for each PDF
Remove-Item Env:MMRAG_CONFIG                  # back to config.yaml
```

In Docker: `LOCAL_EMBED=1 docker compose --profile tools build ingest`, then
`docker compose run --rm -e MMRAG_CONFIG=config.gemma.yaml ingest ingest index "file.pdf"`.

A config is refused against a database whose vectors are another size, so the two cannot be mixed
by mistake. The evaluation judge keeps its own embedding model (`eval.judge_embed_model`), so scores
from both databases are comparable. Keep only what measurably helps (P10); `DROP DATABASE
mmrag_gemma` removes the experiment.

## Embedding on a Kaggle GPU

A local embedding model on a laptop CPU is slow (EmbeddingGemma 2: under one chunk a second on a
2-core laptop). Kaggle's free T4 runs the same model about 20× faster. Only the embedding moves:
PostgreSQL, the PDFs and the app stay on the laptop.

```powershell
mmrag embed bundle [pdf ...]      # parse outputs + code + config (no PDFs); table summaries made here first
mmrag embed push                  # starts `mmrag ingest index --no-db` on a Kaggle GPU
mmrag embed status
mmrag embed pull                  # the vectors it computed
mmrag embed import --index        # merge them into data/cache/embeddings/, then index: database writes only
```

The Kaggle job runs this project's own `ingest index --no-db`, so it embeds exactly the texts a local run
would; `import` only adds vectors the local cache lacks and never replaces it. Uses `KAGGLE_API_TOKEN`
and `caption.kaggle_username`, like captioning. `mmrag ingest index --no-db` also works locally.

## Birth-chart readings by the fine-tuned model (Phase 11)

The chart is computed by code: sidereal positions from NASA JPL's DE440s ephemeris (Skyfield), the
Lahiri ayanamsa, whole-sign houses, nakshatras, dignities, combustion and Vimshottari dashas. The
fine-tuned model [astro-v2-8b](https://huggingface.co/iamaigeek/astro-v2-8b-gguf) only *writes*. It
gets the chart facts and numbered passages from the library, which it cites as [P3]. Code then
drops citations to passages it was not given, and flags any planet position in the text that the
chart contradicts. The chat agent stays on `agent.model`.

The model runs on Modal ([deploy/modal/astro_llm.py](deploy/modal/astro_llm.py)). Point the app at it in `.env`:

```
WRITER_BASE_URL=https://<workspace>--astro-llm-serve.modal.run/v1
WRITER_API_KEY=<the key in the Modal secret astro-llm-key>
MMRAG_EPHEMERIS_DIR=<folder for the 32 MB ephemeris; default data/ephemeris>
```

```powershell
$env:MMRAG_CONFIG = "config.astro.yaml"
mmrag astro chart   --date 1990-05-17 --time 14:30 --tz Asia/Kolkata --lat 28.6139 --lon 77.2090
mmrag astro reading --date 1990-05-17 --time 14:30 --tz Asia/Kolkata --lat 28.6139 --lon 77.2090 [--question "..."]
mmrag astro reading ... --writer openai          # the same reading by agent.model, to compare
mmrag astro ask "What does Saturn in the 7th house give?" [--no-books] [--writer openai]
```

**Judging the readings with Gemini.** `mmrag astro judge` writes the cases in
[eval/astro_cases.yaml](eval/astro_cases.yaml) (every life stage, plus safety probes) and has Gemini review
each reading against the computed chart facts and the passages the writer saw. It lists each error
with a quote and scores six criteria from 1 to 5. It needs `GEMINI_API_KEY` in `.env`. Add
`--writer both` to compare with `agent.model`. The report goes to `data/eval/astro/`.

In the chat app, **🪐 Birth chart reading** in the sidebar does the same. The first reading after a
pause also starts the GPU, which takes a minute or two. The GPU stops 5 minutes after the last request.

## Tests

Development is test-driven where it fits (constitution W5, LLD §10). Install the test tools with `pip install -r requirements-dev.txt`.

| Command | Runs |
|---|---|
| `.venv\Scripts\python -m pytest` | Unit and integration tests. Needs `docker compose up -d`; each integration test uses its own throwaway database. |
| `.venv\Scripts\python -m pytest -m unit` | Fast tests only, no services needed |
| `.venv\Scripts\python -m pytest -m regression` | Re-parses the approved PDFs and compares with `tests/regression/snapshots/` (skipped if the PDFs are absent). After approving a detection change on the review sheet, refresh with `python -m tests.regression.snapshot` |
| `.venv\Scripts\python -m pytest -m live` | Real OpenAI calls (costs a fraction of a cent); run on request |
| `.venv\Scripts\python -m pytest --cov=mmrag` | With a coverage report |
| `docker compose run --rm --entrypoint python ingest -m pytest -q` | The same suite in Linux, for when Windows Smart App Control blocks a library (`tiktoken`) locally |

GitHub runs the suite on every push and pull request (`.github/workflows/tests.yml`).

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
| `src/mmrag/memory/` | Long-term memory: recall, extraction, episodes (Phase 9) |

Database files live in the Docker volume `pgdata`. Back up with `pg_dump`, not by copying files.
