"""astro-v2-8b on Modal: llama.cpp's llama-server (OpenAI-compatible API) on one GPU.

The GGUF (private repo iamaigeek/astro-v2-8b-gguf, 5 GB) is downloaded once into a Modal Volume
with the token from the Modal secret `huggingface` (key HF_TOKEN). Requests need the API key from
the Modal secret `astro-llm-key` (key ASTRO_LLM_API_KEY) as `Authorization: Bearer <key>`.

    python -m modal run deploy/modal/astro_llm.py::download   # once: fetch the model into the volume
    python -m modal deploy deploy/modal/astro_llm.py          # serve; prints the URL
    python deploy/modal/smoke.py --url <URL>                 # one question, from the laptop

The model card asks for the training system prompt (system_v1.txt, also in the volume) and Qwen3
thinking off; the server turns thinking off for every request
(--reasoning off).
"""

from __future__ import annotations

import subprocess

import modal

REPO = "iamaigeek/astro-v2-8b-gguf"
GGUF = "astro-v2-8b.Q4_K_M.gguf"
SHA256 = "4b1898f08f1ef9476ff4221f6eeee6162a2766c4b8dfa6bb70b7f75bfb1ccf5d"
MODELS = "/models"
PORT = 8080
GPU = "L4"  # 24 GB: the 5 GB model plus a 32k-token KV cache (~4.5 GB) fits with room to spare

app = modal.App("astro-llm")
volume = modal.Volume.from_name("astro-llm-models", create_if_missing=True)

# The official llama.cpp CUDA server image; its entrypoint is cleared so Modal can run Python.
server_image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
    .entrypoint([])
)
download_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet]>=0.34")


@app.function(image=download_image, volumes={MODELS: volume}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=3600)
def download() -> None:
    """Fetch the GGUF and the system prompt into the volume and check the GGUF's sha256."""
    import hashlib
    import os

    from huggingface_hub import hf_hub_download

    for name in (GGUF, "system_v1.txt"):
        path = hf_hub_download(REPO, name, local_dir=MODELS, token=os.environ["HF_TOKEN"])
        print("downloaded", path, os.path.getsize(path), "bytes")
    h = hashlib.sha256()
    with open(f"{MODELS}/{GGUF}", "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        raise SystemExit(f"sha256 mismatch: {h.hexdigest()}")
    print("sha256 ok")
    volume.commit()


@app.function(image=server_image, gpu=GPU, volumes={MODELS: volume},
              secrets=[modal.Secret.from_name("astro-llm-key")],
              scaledown_window=300,  # stop the GPU 5 minutes after the last request
              timeout=3600, max_containers=1)
@modal.concurrent(max_inputs=2)
@modal.web_server(PORT, startup_timeout=600)
def serve() -> None:
    import os

    subprocess.Popen([
        "/app/llama-server",
        "--model", f"{MODELS}/{GGUF}",
        "--alias", "astro-v2-8b",
        "--host", "0.0.0.0", "--port", str(PORT),
        "--n-gpu-layers", "999",
        "--ctx-size", "32768", "--parallel", "2",  # 2 slots of 16k: a reading prompt is ~6k tokens
        "--jinja", "--reasoning", "off",  # Qwen3 thinking off, as the model card asks
        "--api-key", os.environ["ASTRO_LLM_API_KEY"],
    ])
    # llama-server opens its port before the model is loaded and answers 503 until then; Modal
    # sends requests as soon as the port opens. Return only once /health says the model is ready.
    import time
    import urllib.request

    deadline = time.monotonic() + 540
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001 - not listening yet, or 503 while loading
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not become ready")

