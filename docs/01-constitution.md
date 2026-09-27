# Project Constitution — Agentic Multimodal RAG

**Status:** Draft v0.6 · **Date:** 2026-09-26 · **Owner:** iamaigeek@gmail.com

This document defines the non-negotiable principles of the project. The technical specification and implementation plan must comply with it. If a design decision conflicts with a principle here, the principle wins, or the constitution is amended explicitly.

---

## 1. Purpose

Build an agentic Retrieval-Augmented Generation system over the project's PDF corpus. It answers questions with **text, original figures from the PDFs, generated charts (bar, pie, line) and data tables**, with every element traceable to its source page and no loss of information.

## 2. Core principles

### P1 — No information loss
- Every piece of content in a PDF (text, embedded images, vector diagrams, tables and scanned pages) must be captured at ingestion. Nothing is silently dropped.
- Anything intentionally skipped, such as decorative logos, repeated headers or icons, is **logged with a reason**, never discarded silently.
- Figures are stored at native resolution. Vector diagrams are rendered at no less than 200 DPI.

### P2 — Provenance for everything
- Every element (text chunk, figure, table, caption, chart data point) carries: **source file, page number, bounding box, element ID**.
- Every block in an answer shows its source. A statement that cannot be cited is not included in the answer.

### P3 — Originals, not imitations
- Images shown to the user are the **original figures from the PDFs**, never AI-generated or redrawn versions.
- The system never generates illustrative images.

### P4 — Charts must be truthful
- A chart is drawn **only from numbers retrieved from the corpus** or computed deterministically from them. The LLM never invents, estimates or "fills in" values.
- Every chart is accompanied by **its underlying data table and source citation**.
- Each value is flagged **exact** (from text or a table) or **estimated** (read visually off a chart image). Charts containing estimated values are labelled "approximate".
- Chart type must match the data:
  - **Pie** only for parts of a whole.
  - **Bar** for comparisons across categories.
  - **Line** for ordered or temporal series.

### P5 — Grounded answers, honest gaps
- The agent answers only from retrieved evidence. If the corpus doesn't contain the answer, it says so plainly.
- The agent must not present general model knowledge as if it came from the documents. If background knowledge is used at all, it is clearly marked as such.

### P6 — Models are replaceable
- The VLM, LLM and embedding model are each accessed through a single interface and selected by configuration.
- Changing a model (for example Qwen-VL to another VLM, or one OpenAI model to another) must not require changes to pipeline logic.

### P7 — Cost- and compute-conscious
- **GPU is used only for VLM captioning at ingestion**, on a temporary or free GPU (Kaggle, Colab or rented). No GPU at query time.
- Query-time reasoning uses the OpenAI API, which is pay-per-use.
- Expensive work (captioning, embeddings) is **cached by content hash** and never repeated for unchanged content.

### P8 — Reproducible, incremental ingestion
- Ingestion is deterministic, and every stage can be re-run on its own.
- Adding a new PDF processes only that PDF.
- Each stage writes inspectable intermediate outputs (JSON/JSONL, PNG) so any step can be audited by hand.

### P8a — One source of truth, consistent data
- A single database holds all metadata, captions, tables, vectors and the keyword index. There is no second store that can drift out of sync.
- Each document is written **atomically**: all of its data is committed together, or none of it is. Search never returns results from a half-ingested document, or from a mix of old and new versions.
- The database can always be rebuilt from the source PDFs plus cached intermediate outputs.

### P9 — Simplicity first
- Prefer plain Python and direct SDK calls over heavy frameworks, unless a framework removes significant complexity.
- Build the smallest working version end to end first, then improve quality.

### P10 — Measured, not assumed
- Quality claims are backed by an evaluation set, scored with RAGAS plus project-specific checks (charts, figures, citations, refusals).
- Changes to prompts, models or retrieval are compared against the previous baseline before being kept.

### P11 — Authenticated, accountable access
- Only signed-in users can use the application. Accounts are created by an administrator; there is no public sign-up.
- Passwords are stored only as strong one-way hashes (Argon2id). They are never logged, displayed or stored in plain text.
- A failed sign-in never reveals whether an account exists. Repeated failures lock the account temporarily.
- Every question and every sign-in event is attributable to a user. Each user has daily usage limits that protect the API budget.

### P12 — Observable by default
- Every question and every ingestion run is **traced end to end**. Each plan, agent round, tool call, model call, database query and validation step can be inspected afterwards.
- Telemetry follows an open standard (OpenTelemetry), so the viewing tool can be changed without code changes.
- Telemetry never breaks the application and never contains secrets or passwords. It records user content only where explicitly allowed, and is kept only as long as the retention policy says.

## 3. Working agreements (human ↔ AI assistant)

- **W1 — No code without permission.** The AI assistant does not write or run code until the owner explicitly approves the specific phase or task.
- **W2 — Roadmap before build.** Each phase starts with a short plan. Work begins only after owner approval.
- **W3 — Phase gates.** Each phase ends with a demo or evidence of its acceptance criteria and owner sign-off before the next phase begins.
- **W4 — Transparency.** Failures, skipped steps, cost overruns and known limitations are reported plainly.
- **W5 — Test-driven development, where it fits.** Code is proven by automated tests, not by manual checks.
  - **Deterministic logic** (config, IDs, chunking, search scoring, `compute`, chart validation, the answer validator, citations, sign-in and sessions, limits, database writes) is built **test-first**: a failing test from the acceptance criteria, then code until it passes, then refactoring with all tests green.
  - **Exploratory parts** (PDF figure and table detection) are tuned with the owner on the review sheet first. The approved results are then frozen as regression tests.
  - **LLM answer quality** is judged by the RAGAS evaluation (P10), not by unit tests.

  The full suite runs after every change. Phase summaries report test counts and results.

## 4. Out of scope (v1)

- Generating new images or illustrations.
- Real-time collaboration.
- Self-service sign-up, email-based password reset, single sign-on and multi-factor authentication. v1 uses admin-created email + password accounts (P11); these may be added later.
- Fine-tuning any model.
- Non-PDF sources (web pages, DOCX, video). These may be added later without violating P6.

## 5. Amendments

Changes to this document require the owner's explicit approval and are recorded below.

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-09-26 | Initial draft |
| 0.2 | 2026-09-26 | Added P8a (one source of truth, atomic writes per document), following the owner's decision to use PostgreSQL for all storage |
| 0.3 | 2026-09-26 | Added P11 (authenticated, accountable access) after the owner chose email + password login; out-of-scope list updated |
| 0.4 | 2026-09-26 | Added P12 (observable by default, OpenTelemetry) at the owner's request |
| 0.5 | 2026-09-26 | P10 names RAGAS as the evaluation framework, at the owner's request |
| 0.6 | 2026-09-27 | Added W5 (test-driven development where it fits: TDD for deterministic logic, frozen regression tests for exploratory parsing, RAGAS for LLM quality), at the owner's request |
