"""astro-v2-8b on Modal: llama.cpp's llama-server (OpenAI-compatible API) on one GPU.

The GGUF (private repo iamaigeek/astro-v2-8b-gguf, 5 GB) is downloaded once into a Modal Volume
with the token from the Modal secret `huggingface` (key HF_TOKEN). Requests need the API key from
the Modal secret `astro-llm-key` (key ASTRO_LLM_API_KEY) as `Authorization: Bearer <key>`.

    python -m modal run deploy/modal/astro_llm.py::download   # once: fetch the model into the volume
    python -m modal deploy deploy/modal/astro_llm.py          # serve; prints the URL
    python -m modal run deploy/modal/astro_llm.py::smoke --url <URL>

The model card asks for the training system prompt (system_v1.txt, also in the volume) and Qwen3
thinking off; the server turns thinking off for every request.
"""

from __future__ import annotations

import subprocess

import modal

REPO = "iamaigeek/astro-v2-8b-gguf"
GGUF = "astro-v2-8b.Q4_K_M.gguf"
SHA256 = "4b1898f08f1ef9476ff4221f6eeee6162a2766c4b8dfa6bb70b7f75bfb1ccf5d"
MODELS = "/models"
PORT = 8080
GPU = "L4"  # 24 GB: the 5 GB model plus a full 16k context fits with room to spare

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
@modal.concurrent(max_inputs=4)
@modal.web_server(PORT, startup_timeout=600)
def serve() -> None:
    import os

    subprocess.Popen([
        "/app/llama-server",
        "--model", f"{MODELS}/{GGUF}",
        "--alias", "astro-v2-8b",
        "--host", "0.0.0.0", "--port", str(PORT),
        "--n-gpu-layers", "999",
        "--ctx-size", "16384", "--parallel", "4",  # 4 slots of 4k tokens each
        "--jinja", "--chat-template-kwargs", '{"enable_thinking": false}',
        "--api-key", os.environ["ASTRO_LLM_API_KEY"],
    ])


@app.local_entrypoint()
def smoke(url: str, question: str = "What does Jupiter in the 10th house indicate?") -> None:
    """Ask one question through the deployed endpoint, with the training system prompt."""
    import json
    import os
    import urllib.request
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    system = (Path(__file__).with_name("system_v1.txt")).read_text(encoding="utf-8")
    body = {"model": "astro-v2-8b", "temperature": 0.3, "max_tokens": 400,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}]}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['ASTRO_LLM_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=900) as r:  # the first call waits for a cold start
        reply = json.load(r)
    print(reply["choices"][0]["message"]["content"])
    print(json.dumps(reply.get("usage"), indent=1))
