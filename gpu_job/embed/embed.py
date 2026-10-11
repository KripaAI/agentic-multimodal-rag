"""GPU embedding job (Kaggle): the project's own `mmrag ingest index --no-db`, run on a GPU.

Standalone like the captioning job. The bundle (a Kaggle dataset) holds project.bundle (the mmrag
source, the config files, and the parse outputs of the PDFs: elements, tables, captions,
transcripts, table summaries), manifest.json ({pdf name: {doc_id, sha256}}, so the PDFs themselves
are not uploaded) and run.json. The job writes the embedding-cache lines it computed to
<out>/embeddings.jsonl; on the laptop `mmrag embed import` merges them into the local cache, and
`mmrag ingest index` then finds every vector there and only writes to PostgreSQL.

    python embed.py                       # on Kaggle
    EMBED_BUNDLE=dir EMBED_OUT=dir EMBED_SKIP_INSTALL=1 python embed.py   # a local dry run
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# What `mmrag.cli ingest index` imports with telemetry off; the full requirements.txt (Streamlit,
# LangGraph, ...) is not needed here and is slow to install on Kaggle's Python.
PACKAGES = ["pyyaml", "python-dotenv", "pydantic>=2", "openai", "psycopg[binary]>=3.1", "pgvector", "tiktoken",
            "opentelemetry-api", "opentelemetry-sdk"]
# Upgraded, not just installed: Kaggle's preinstalled transformers did not know EmbeddingGemma 2's
# architecture (embedding_gemma2). The versions are those the laptop runs the model with.
MODEL_PACKAGES = ["sentence-transformers>=6.1", "transformers>=5.19"]


def find_bundle() -> Path:
    if os.environ.get("EMBED_BUNDLE"):
        return Path(os.environ["EMBED_BUNDLE"])
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for candidate in [here, *map(Path, glob.glob("/kaggle/input/**/", recursive=True))]:
        if (candidate / "run.json").is_file() and (candidate / "project.bundle").is_file():
            return candidate
    raise SystemExit("bundle not found: no run.json + project.bundle next to the script or under /kaggle/input")


def use_kaggle_hf_token() -> None:
    """Sets HF_TOKEN from the Kaggle secret of that name, if the kernel has one attached (Add-ons >
    Secrets in the Kaggle editor). The models are not gated: the token only lifts the Hub's rate
    limit for anonymous downloads. It is never put in the bundle, which is uploaded as a dataset."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001 - not on Kaggle, or no such secret attached: download anonymously
        pass


def main() -> int:
    use_kaggle_hf_token()
    bundle = find_bundle()
    run = json.loads((bundle / "run.json").read_text(encoding="utf-8"))
    out = Path(os.environ.get("EMBED_OUT", "/kaggle/working"))
    project = out / "project"
    if project.exists():
        shutil.rmtree(project)
    with zipfile.ZipFile(bundle / "project.bundle") as z:
        z.extractall(project)
    shutil.copyfile(bundle / "manifest.json", project / "manifest.json")

    if not os.environ.get("EMBED_SKIP_INSTALL"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *MODEL_PACKAGES], check=True)
        # sentence-transformers may bring another torch; the image's torchaudio is then built for a
        # different CUDA version and breaks the model load (the same fix as the captioning job).
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)

    # Runs on the laptop's config, with telemetry off and a GPU-sized batch; nothing touches a database.
    (project / "config.kaggle.yaml").write_text(
        f"extends: {run['config']}\n\npaths:\n  data_dir: data\n\nobservability:\n  enabled: false\n  log_file: null\n\n"
        f"embed:\n  batch_size: {run.get('batch_size', 32)}\n", encoding="utf-8")
    env = {**os.environ, "MMRAG_CONFIG": str(project / "config.kaggle.yaml"), "PYTHONPATH": str(project / "src"),
           "DATABASE_URL": "postgresql://unused:unused@localhost:1/unused", "OPENAI_API_KEY": "",
           "PYTHONUTF8": "1"}
    started = time.monotonic()
    r = subprocess.run([sys.executable, "-m", "mmrag.cli", "ingest", "index", "--no-db",
                        "--manifest", str(project / "manifest.json")], cwd=project, env=env)

    cache = sorted((project / "data" / "cache" / "embeddings").glob("*.jsonl"))
    lines = [line for f in cache for line in f.read_text(encoding="utf-8").split("\n") if line.strip()]
    (out / "embeddings.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    try:
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - the report is informative only
        device = "unknown"
    report = {"task": "embed", "exit_code": r.returncode, "documents": len(json.loads(
        (project / "manifest.json").read_text(encoding="utf-8"))), "vectors": len(lines),
        "cache_files": [f.name for f in cache], "seconds": round(time.monotonic() - started, 1), "device": device}
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    shutil.rmtree(project, ignore_errors=True)  # only embeddings.jsonl and summary.json are outputs
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
