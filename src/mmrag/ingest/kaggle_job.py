"""Run the captioning job on Kaggle's GPU from the laptop (plan Phase 2, task 3).

The bundle goes up as a private Kaggle dataset; the job script goes up as a private
Kaggle kernel with GPU and internet (for the model download) switched on; results come
back with `kaggle kernels output`. Uses the `kaggle` CLI and KAGGLE_API_TOKEN from .env.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mmrag.config import PROJECT_ROOT, Settings
from mmrag.fsutil import empty_dir

JOB_SCRIPT = PROJECT_ROOT / "gpu_job" / "caption" / "caption.py"


def dataset_slug(doc_id: str, task: str = "caption") -> str:
    # Not "mmrag-caption-...": after a failed create, Kaggle kept those addresses reserved but empty.
    return f"mmrag-bundle-{doc_id}" if task == "caption" else f"mmrag-pages-{doc_id}"


def kernel_slug(doc_id: str, task: str = "caption") -> str:
    # Must differ from the dataset's: Kaggle titles are unique per user. Transcription has its own
    # kernel, so it never replaces a document's captioning job or its outputs.
    return f"mmrag-captioner-{doc_id}" if task == "caption" else f"mmrag-transcriber-{doc_id}"


def dataset_metadata(user: str, doc_id: str, task: str = "caption") -> dict:
    title = f"mmrag bundle {doc_id}" if task == "caption" else f"mmrag pages {doc_id}"
    return {"title": title, "id": f"{user}/{dataset_slug(doc_id, task)}", "licenses": [{"name": "other"}]}


def kernel_metadata(user: str, doc_id: str, task: str = "caption", gpu: bool = True) -> dict:
    """Private script kernel reading the bundle dataset, on a GPU machine unless the job only runs
    a CPU engine (tesseract), which would waste GPU quota. Internet is on for model downloads."""
    return {
        "id": f"{user}/{kernel_slug(doc_id, task)}",
        "title": kernel_slug(doc_id, task),
        "code_file": JOB_SCRIPT.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        **({"machine_shape": "NvidiaTeslaT4"} if gpu else {}),
        "dataset_sources": [f"{user}/{dataset_slug(doc_id, task)}"],
        "competition_sources": [],
        "kernel_sources": [],
    }


def _kaggle(*args: str) -> subprocess.CompletedProcess:
    # UTF-8 mode: on Windows the CLI otherwise crashes printing the job log ('charmap' codec).
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    # The CLI exits 0 on some failures and only prints them ("Dataset creation error: ...", or
    # "Kernel push error: Maximum batch GPU session count of 2 reached.").
    if r.returncode != 0 or any(m in r.stdout.lower() for m in ("creation error", "push error")):
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r


def _wait_until_ready(dataset: str, timeout_s: int = 1800) -> None:
    """A new dataset version is processed for a while before a kernel can mount it (about 20
    minutes for the 13 MB Post-Training bundle in Phase 5)."""
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if "ready" in _kaggle("datasets", "status", dataset).stdout.lower():
                return
        except RuntimeError as e:
            if "403" not in str(e):  # a brand-new dataset answers 403 until Kaggle has registered it
                raise
        time.sleep(10)
    raise RuntimeError(f"dataset {dataset} not ready after {timeout_s}s")


def push(bundle: Path, doc_id: str, settings: Settings, workdir: Path, task: str = "caption",
         gpu: bool = True) -> str:
    """Upload (or version) the bundle dataset, then push and start the kernel. Returns
    the kernel reference `user/slug`."""
    user = settings.caption.kaggle_username
    (bundle / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(user, doc_id, task), indent=1),
                                                  encoding="utf-8")
    exists = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status",
                             f"{user}/{dataset_slug(doc_id, task)}"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace").returncode == 0
    if exists:
        _kaggle("datasets", "version", "-p", str(bundle), "-m", "new bundle", "--dir-mode", "zip")
    else:
        _kaggle("datasets", "create", "-p", str(bundle), "--dir-mode", "zip")
    _wait_until_ready(f"{user}/{dataset_slug(doc_id, task)}")

    kernel_dir = empty_dir(workdir / "kernel")
    shutil.copyfile(JOB_SCRIPT, kernel_dir / JOB_SCRIPT.name)
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(user, doc_id, task, gpu), indent=1),
                                                     encoding="utf-8")
    _kaggle("kernels", "push", "-p", str(kernel_dir))
    return f"{user}/{kernel_slug(doc_id, task)}"


def status(doc_id: str, settings: Settings, task: str = "caption") -> str:
    return _kaggle("kernels", "status",
                   f"{settings.caption.kaggle_username}/{kernel_slug(doc_id, task)}").stdout.strip()


def pull(doc_id: str, settings: Settings, out: Path, task: str = "caption") -> Path:
    """Download the kernel's output files (captions or transcripts, reports, log) into `out`."""
    out.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", f"{settings.caption.kaggle_username}/{kernel_slug(doc_id, task)}",
            "-p", str(out), "-o")
    return out
