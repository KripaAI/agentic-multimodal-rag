"""Ask the deployed astro-v2-8b one question, with the training system prompt.

A plain script, not a Modal entrypoint: `modal run` would also start a temporary copy of the app.

    python deploy/modal/smoke.py --url https://iamaigeek--astro-llm-serve.modal.run
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://iamaigeek--astro-llm-serve.modal.run")
    parser.add_argument("--question", default="What does Jupiter in the 10th house indicate?")
    parser.add_argument("--max-tokens", type=int, default=600)
    args = parser.parse_args()
    load_dotenv(HERE.parents[1] / ".env")
    body = {"model": "astro-v4-8b", "temperature": 0.3, "max_tokens": args.max_tokens,
            "messages": [{"role": "system", "content": (HERE / "system_v1.txt").read_text(encoding="utf-8")},
                         {"role": "user", "content": args.question}]}
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['ASTRO_LLM_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=900) as r:  # the first call waits for a cold start
        reply = json.load(r)
    print(reply["choices"][0]["message"]["content"])
    print(json.dumps(reply.get("usage"), indent=1))


if __name__ == "__main__":
    main()
