# Operations guide

How to run, maintain and deploy the system once it is set up (Phase 8). Setup itself is in the [README](../README.md).

## 1. The PDF library

Each command touches one PDF only. The other PDFs are never re-processed.

| Task | Command |
|---|---|
| See what is indexed | `mmrag doc list` |
| Add a new PDF | `mmrag doc add path\to\new.pdf` |
| Replace a PDF with a new version | `mmrag doc replace "old name.pdf" path\to\new-version.pdf` |
| Remove a PDF | `mmrag doc remove "name.pdf" --yes` |

`add` and `replace` parse the PDF and print the remaining steps:
1. **Check the review sheet** that the parse step printed.
2. **Describe the figures on the GPU:** `mmrag caption bundle`, then `push`, `pull` and `import`, each followed by the file name.
3. **Index it:** `mmrag ingest index "name.pdf"`.

**On Windows with Smart App Control on,** indexing fails because Windows blocks the `tiktoken` library ("An Application Control policy has blocked this file"). Run ingestion commands in the Linux container instead. Your security setting stays on, and the project folder and database are shared with the container:

```powershell
docker compose --profile tools build ingest            # once (and after requirements.txt changes)
docker compose run --rm ingest doc add data/new.pdf     # the path must be inside the project folder
docker compose run --rm ingest ingest index "new.pdf"
```

Any `mmrag` command works after `docker compose run --rm ingest`. The chat app, search and `mmrag ask` don't need `tiktoken` and run on Windows as usual.

**Replace is safe.** Search keeps answering from the old version until step 3 swaps the new one in, in a single database transaction. The old file is kept in `data/pdfs/replaced/`.

**Remove** deletes the document and all its search data in one cascading delete. Its PDF moves to `data/pdfs/removed/`.

**Figures can be excluded per PDF** in `config.yaml`, under `enrich.skip_figures_for`. This is used for the Post-Training guide, whose captions don't match its pictures (see the Phase 6 results).

## 2. Backups

| Task | Command |
|---|---|
| Back up | `mmrag db backup` (writes `data/backups/mmrag-<time>.dump`) |
| Prove a backup restores correctly | `mmrag db verify-restore data\backups\<file>.dump` |
| Restore | `mmrag db restore <file>.dump --into mmrag_restored` |

- **Restores never overwrite a database.** To switch to a restored copy, point `DATABASE_URL` in `.env` at the new database name and restart the app.
- **`verify-restore` proves a backup is usable.** It restores into a scratch database, runs the 25 retrieval test queries against both copies, checks the results are identical, then drops the scratch copy. On 2026-10-01 it returned 25/25 identical.
- **Keep backups out of git and off shared folders.** They contain user accounts (password hashes only) and question history.
- **Schedule** a daily backup and a weekly `verify-restore`, for example with Windows Task Scheduler or cron. Copy the dumps to a second disk or cloud storage.
- **Where the tools come from:** `pg_dump` and `pg_restore` are used from PATH when installed, as with a managed database or CI. Otherwise they run inside the docker compose `db` container.

## 3. Accounts and the sign-in audit

| Task | Command |
|---|---|
| Create an account (temporary password, shown once) | `mmrag user add someone@example.com [--role admin]` |
| Forgotten password | `mmrag user reset-password someone@example.com` |
| Lock out a user / let them back in | `mmrag user disable …` / `mmrag user enable …` |
| Clear a lockout after 5 failed sign-ins | `mmrag user unlock …` |
| List accounts | `mmrag user list` |
| Review sign-in events | `mmrag user events --days 7` |

**Admin password recovery.** There is no e-mail reset; recovery is by design an operator action on the server. Anyone with shell access to the server runs `mmrag user reset-password <admin email>`. That revokes every session of that account and prints a one-time temporary password, which must be changed at the next sign-in. Server shell access is therefore the root of trust, so protect it.

**Weekly audit review** with `mmrag user events --days 7`. Look for:
- many `login_failure` events for one email or from one IP (password guessing);
- `lockout` events;
- `user_created`, `password_reset` or `user_enabled` events you didn't do yourself.

## 4. Monitoring and alerts

- **Retention cleanup:** `mmrag obs cleanup --yes`. Run it weekly. It deletes:
  - query history older than `retention.query_log_days` (90);
  - sign-in events older than `retention.auth_events_days` (365);
  - sessions that ended more than 30 days ago;
  - the saved conversation state of threads whose history is gone.
- **Alerts:** `mmrag obs alerts` exits with code 1 when any of these hold:
  - today's total cost is over `alerts.daily_cost_usd`;
  - more than `alerts.dropped_rate` of the last hour's answers had unverified parts removed (only judged once there are at least `alerts.min_questions`);
  - there are more than `alerts.failed_sign_ins_per_hour` failed sign-ins in the last hour.

  Run it every 15 minutes from Task Scheduler or cron, and make a non-zero exit send you a notification (an e-mail action, or a webhook to your phone).
- **Traces in production** go through the OpenTelemetry Collector (`deploy/otel-collector.yaml`).
  - The collector keeps **every error and every slow (over 20 s) trace**, plus 10% of the rest, and forwards them to Phoenix.
  - Run it with `otelcol-contrib --config deploy/otel-collector.yaml`.
  - The app exports in the background. If the collector or Phoenix is down, answers are not affected; a test checks this.
- **Admin metrics page:** in the app, the admin sidebar has 📊 Admin metrics: questions per day, slowest answers linked to traces, cost per user, and failed sign-ins.

## 5. Deploying with HTTPS

The app must only be reached over HTTPS. Streamlit itself listens on `127.0.0.1:8501` only (`.streamlit/config.toml`).

1. **Prepare the server:** a server with a domain name pointing at it, and ports 80 and 443 open.
2. **Configure Caddy:** edit `deploy/Caddyfile` and replace `rag.example.com` with your domain.
3. **Start the proxy:** run `caddy run --config deploy/Caddyfile`. Caddy obtains and renews the TLS certificate automatically and adds security headers.
4. **Change these settings** in `config.yaml` (or a separate file named in `MMRAG_CONFIG`):

   | Setting | Production value | Why |
   |---|---|---|
   | `auth.trust_proxy_headers` | `true` | Sign-in throttling uses the client IP that Caddy passes on. Set it **only** behind the proxy. |
   | `observability.environment` | `prod` | |
   | `observability.capture_content` | `false` | No prompts or document text in traces |
   | `observability.otlp_traces_endpoint` | `http://127.0.0.1:4318/v1/traces` | Send traces to the collector |
   | `auth.limits_timezone` | your zone, e.g. `Asia/Kolkata` | Daily limits reset at your midnight |

5. **Run both services:** `streamlit run app/ui.py` and the collector. Use a service manager (NSSM on Windows, systemd on Linux) so they restart after a reboot.

## 6. Changing the embedding model (blue-green)

Changing `embed.model` or `embed.dims` makes every stored vector incompatible (spec §6.5). Do it without downtime:
1. **Back up** with `mmrag db backup`.
2. **Create a second database:** `mmrag db restore <dump> --into mmrag_green`.
3. **Prepare the new configuration:** copy `config.yaml` to `config.green.yaml` and set the new model and dimensions. Above 1536 dimensions, a migration must change the `vector(1536)` column first.
4. **Re-index into the green database:** set `MMRAG_CONFIG=config.green.yaml` and point `DATABASE_URL` at `mmrag_green`, then run `mmrag ingest index` for every PDF. Parse output and captions are reused; only embeddings are recomputed.
5. **Check it:** `mmrag search-report` and `mmrag eval retrieval` against green. Switch only if the scores hold up.
6. **Switch:** point the app's `DATABASE_URL` at `mmrag_green` and restart. Keep the old database until you are sure.

## 7. Moving to managed PostgreSQL

Any managed PostgreSQL 16 with the **pgvector** extension works (for example Azure Database for PostgreSQL, AWS RDS, Google Cloud SQL, Supabase or Neon).
1. **Create** the database and enable pgvector (`CREATE EXTENSION vector`). Migration 0001 also does this, if the user is allowed to.
2. **Back up and restore:** `mmrag db backup` locally, then `pg_restore --no-owner -d <managed url> <dump>`, using a local PostgreSQL 16 client.
3. **Point at it:** set `DATABASE_URL` in `.env` to the managed database's address, with `?sslmode=require`.
4. **Verify:** run `mmrag db migrate` (it should report up to date), then `mmrag search-report`.

## 8. Continuous integration

`.github/workflows/tests.yml` runs the unit and integration tests on every push and pull request. It uses a temporary PostgreSQL 16 + pgvector, no OpenAI key, and costs nothing.
- **Tests that need the source PDFs** (regression) are skipped, because the PDFs are not in git.
- **Paid tests** (`-m live`) are run manually.
- **Results** are in the repository's **Actions** tab and on each pull request.
