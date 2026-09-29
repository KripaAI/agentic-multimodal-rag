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


def dataset_slug(doc_id: str) -> str:
    # Not "mmrag-caption-...": after a failed create, Kaggle kept those addresses reserved but empty.
    return f"mmrag-bundle-{doc_id}"


def kernel_slug(doc_id: str) -> str:
    return f"mmrag-captioner-{doc_id}"  # must differ from the dataset's: Kaggle titles are unique per user


def dataset_metadata(user: str, doc_id: str) -> dict:
    return {"title": f"mmrag bundle {doc_id}", "id": f"{user}/{dataset_slug(doc_id)}",
            "licenses": [{"name": "other"}]}


def kernel_metadata(user: str, doc_id: str) -> dict:
    """Private GPU script kernel reading the bundle dataset. Internet is on only so the
    job can download the model weights from Hugging Face."""
    return {
        "id": f"{user}/{kernel_slug(doc_id)}",
        "title": kernel_slug(doc_id),
        "code_file": JOB_SCRIPT.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{user}/{dataset_slug(doc_id)}"],
        "competition_sources": [],
        "kernel_sources": [],
    }


def _kaggle(*args: str) -> subprocess.CompletedProcess:
    # UTF-8 mode: on Windows the CLI otherwise crashes printing the job log ('charmap' codec).
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    # The CLI exits 0 on some failures and only prints them (e.g. "Dataset creation error: ...").
    if r.returncode != 0 or "creation error" in r.stdout.lower():
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r


def _wait_until_ready(dataset: str, timeout_s: int = 300) -> None:
    """A new dataset version is processed for a while before a kernel can mount it."""
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


def push(bundle: Path, doc_id: str, settings: Settings, workdir: Path) -> str:
    """Upload (or version) the bundle dataset, then push and start the kernel. Returns
    the kernel reference `user/slug`."""
    user = settings.caption.kaggle_username
    (bundle / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(user, doc_id), indent=1), encoding="utf-8")
    exists = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", f"{user}/{dataset_slug(doc_id)}"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace").returncode == 0
    if exists:
        _kaggle("datasets", "version", "-p", str(bundle), "-m", "new bundle", "--dir-mode", "zip")
    else:
        _kaggle("datasets", "create", "-p", str(bundle), "--dir-mode", "zip")
    _wait_until_ready(f"{user}/{dataset_slug(doc_id)}")

    kernel_dir = empty_dir(workdir / "kernel")
    shutil.copyfile(JOB_SCRIPT, kernel_dir / JOB_SCRIPT.name)
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(user, doc_id), indent=1),
                                                     encoding="utf-8")
    _kaggle("kernels", "push", "-p", str(kernel_dir))
    return f"{user}/{kernel_slug(doc_id)}"


def status(doc_id: str, settings: Settings) -> str:
    return _kaggle("kernels", "status", f"{settings.caption.kaggle_username}/{kernel_slug(doc_id)}").stdout.strip()


def pull(doc_id: str, settings: Settings, out: Path) -> Path:
    """Download the kernel's output files (captions, reports, log) into `out`."""
    out.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", f"{settings.caption.kaggle_username}/{kernel_slug(doc_id)}", "-p", str(out), "-o")
    return out
