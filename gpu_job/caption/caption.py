"""VLM figure captioning job (LLD §3.4, spec §6.2). Runs on Kaggle/Colab, not locally.

Standalone on purpose: it imports nothing from `mmrag`, so a bare GPU notebook can run
it. The bundle (a Kaggle dataset) holds jobs.jsonl, images/, the prompt and run.json.

    python caption.py                 # every model listed in run.json, one subprocess each
    python caption.py --model 3b      # one model path

Outputs go to <out>/<model_path>/: captions.jsonl, done_ids.txt, run_report.json.
Each model runs in its own process so GPU memory is fully released between models.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ---------------------------------------------------------------- schema
# Identical to mmrag/ingest/captions.py (a unit test compares the JSON schemas).

MAX_VISIBLE_TEXT = 80  # labels in the densest pilot figure: about 40

FigureType = Literal["diagram", "flowchart", "chart", "table_image", "screenshot", "photo", "equation", "decorative"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataPoint(_Model):
    label: str
    value: float
    flag: Literal["exact", "estimated"]  # exact: printed on the figure; estimated: read off a bar or line


class Series(_Model):
    name: str
    points: list[DataPoint]


class ChartData(_Model):
    chart_kind: Literal["bar", "line", "pie", "scatter", "other"]
    x_label: str | None = None
    y_label: str | None = None
    unit: str | None = None
    series: list[Series]


class TableData(_Model):
    columns: list[str]
    rows: list[list[str]]


class ExtractedData(_Model):
    chart: ChartData | None = None
    table: TableData | None = None


class FigureCaption(_Model):
    """The VLM's answer for one figure (spec §5.2, without `model_id`, which the job adds)."""

    figure_type: FigureType
    short_caption: str = Field(min_length=1)
    detailed_description: str = Field(min_length=1)
    visible_text: list[str] = Field(max_length=MAX_VISIBLE_TEXT)  # a cap stops label-repeating loops
    extracted_data: ExtractedData | None = None
    keywords: list[str]
    confidence: Literal["high", "medium", "low"]


MODELS = {
    "awq-7b": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "nf4-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}
MAX_NEW_TOKENS = 2048
# Pilot v2: a repetition penalty (1.05) blanked table cells whose text the model had
# already written in visible_text, so none is used; the visible_text cap stops loops.
REPETITION_PENALTY = 1.0
MIN_PIXELS = 256 * 28 * 28

# ---------------------------------------------------------------- prompt and parsing


def build_prompt(template: str, job: dict) -> str:
    schema = json.dumps(FigureCaption.model_json_schema(), separators=(",", ":"))
    return template.format(
        source_file=job["source_file"],
        page=job["page"],
        section_path=" > ".join(job["section_path"]) or "(none)",
        pdf_caption=job["pdf_caption"] or "(none)",
        context_before=job["context_before"] or "(none)",
        context_after=job["context_after"] or "(none)",
        figure_refs=" | ".join(job["figure_refs"]) or "(none)",
        schema=schema,
    )


def _repair(raw: str) -> str:
    """Common model slips: Markdown fences, chatter around the object, trailing commas, curly quotes."""
    text = re.sub(r"```(?:json)?", "", raw).replace("“", '"').replace("”", '"')
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return text
    return re.sub(r",\s*([}\]])", r"\1", text[start:end + 1])


def parse_caption(raw: str) -> tuple[dict | None, str | None]:
    """Validated caption dict, or (None, error). Tries the raw text, then a repaired copy."""
    error = "empty reply"
    for candidate in (raw, _repair(raw)):
        try:
            return FigureCaption.model_validate(json.loads(candidate)).model_dump(), None
        except json.JSONDecodeError as e:
            error = f"invalid JSON: {e}"
        except ValidationError as e:
            error = f"schema: {e}"
    return None, error


# ---------------------------------------------------------------- the run loop


def run_jobs(jobs, backend, bundle: Path, out: Path, model_path: str, prompt_version: str, template: str,
             batch_size: int = 4) -> dict:
    """Caption every job not already in out/done_ids.txt. Invalid output is retried once,
    then written with status needs_review and the raw text. Returns the run report."""
    out.mkdir(parents=True, exist_ok=True)
    done_file = out / "done_ids.txt"
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.is_file() else set()
    todo = [j for j in jobs if j["element_id"] not in done]
    started = time.monotonic()

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        items = [(j, build_prompt(template, j), bundle / j["image"]) for j in batch]
        t0 = time.monotonic()
        replies = backend.generate(items)
        parsed = [parse_caption(r) for r in replies]
        retry = [k for k, (cap, _) in enumerate(parsed) if cap is None]
        if retry:
            for k, raw in zip(retry, backend.generate([items[k] for k in retry])):
                replies[k] = raw
                parsed[k] = parse_caption(raw)
        per_item = (time.monotonic() - t0) / len(batch)
        with (out / "captions.jsonl").open("a", encoding="utf-8") as f, done_file.open("a", encoding="utf-8") as d:
            for job, raw, (cap, err) in zip(batch, replies, parsed):
                f.write(json.dumps({
                    "element_id": job["element_id"], "image_hash": job["image_hash"],
                    "status": "ok" if cap else "needs_review", "caption": cap, "error": err,
                    "raw": None if cap else raw, "model_path": model_path, "model_id": backend.model_id,
                    "prompt_version": prompt_version, "seconds": round(per_item, 2),
                }, ensure_ascii=False) + "\n")
                d.write(job["element_id"] + "\n")
        print(f"[{model_path}] {min(i + batch_size, len(todo))}/{len(todo)} done", flush=True)

    records = [json.loads(line) for line in (out / "captions.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = sum(r["status"] == "ok" for r in records)
    report = {
        "model_path": model_path, "model_id": backend.model_id, "prompt_version": prompt_version,
        "jobs": len(records), "ok": ok, "needs_review": len(records) - ok,
        "validity_rate": ok / len(records) if records else 0.0,
        "seconds_this_run": round(time.monotonic() - started, 1),
        "seconds_per_figure": round(sum(r["seconds"] for r in records) / len(records), 2) if records else 0.0,
        "gpu_memory_mb": gpu_memory_mb(),
    }
    (out / "run_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def gpu_memory_mb() -> list[int]:
    """Memory in use on each GPU right now (nvidia-smi), or [] without a GPU."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return [int(x) for x in r.stdout.split()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


# ---------------------------------------------------------------- model backends (GPU only)


def _pip(*packages: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def _data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


class TransformersBackend:
    """Fallback A (nf4-7b: 4-bit NF4 via bitsandbytes) and fallback B (3b: fp16). T4 has no bf16."""

    def __init__(self, model_path: str, max_pixels: int):
        _pip("-U", "transformers", "accelerate", "bitsandbytes")
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass

        self.model_id = MODELS[model_path]
        quant = None
        if model_path == "nf4-7b":
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = ModelClass.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto",
                                                quantization_config=quant)
        self.processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=MIN_PIXELS, max_pixels=max_pixels)
        self.torch = torch

    def generate(self, items):
        from PIL import Image

        replies = []
        for _, prompt, image_path in items:  # one at a time: predictable memory on a T4
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                             repetition_penalty=REPETITION_PENALTY)
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            replies.append(self.processor.decode(new_tokens, skip_special_tokens=True))
        return replies


class VLLMBackend:
    """Primary: AWQ 7B on vLLM in fp16, with the JSON schema enforced during decoding."""

    def __init__(self, model_path: str, max_pixels: int):
        _pip("vllm", "qwen-vl-utils")
        from vllm import LLM, SamplingParams

        self.model_id = MODELS[model_path]
        self.llm = LLM(model=self.model_id, dtype="float16", quantization="awq", max_model_len=8192,
                       gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 1}, enforce_eager=True,
                       mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": max_pixels})
        schema = FigureCaption.model_json_schema()
        try:  # the structured-output API was renamed between vLLM releases
            from vllm.sampling_params import StructuredOutputsParams
            self.params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, repetition_penalty=REPETITION_PENALTY,
                                         structured_outputs=StructuredOutputsParams(json=schema))
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            self.params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, repetition_penalty=REPETITION_PENALTY,
                                         guided_decoding=GuidedDecodingParams(json=schema))

    def generate(self, items):
        conversations = [[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            {"type": "text", "text": prompt},
        ]}] for _, prompt, image_path in items]
        return [o.outputs[0].text for o in self.llm.chat(conversations, self.params, use_tqdm=False)]


def load_model(model_path: str, max_pixels: int):
    if model_path == "awq-7b":
        return VLLMBackend(model_path, max_pixels)
    return TransformersBackend(model_path, max_pixels)


# ---------------------------------------------------------------- entry point


def find_bundle() -> Path:
    """The bundle folder: next to this script (local/Colab) or under /kaggle/input."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for candidate in [here, *map(Path, glob.glob("/kaggle/input/**/", recursive=True))]:
        if (candidate / "run.json").is_file():
            return candidate
    raise SystemExit("bundle not found: no run.json next to the script or under /kaggle/input")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), help="run one model path (default: all in run.json)")
    parser.add_argument("--out", default=os.environ.get("CAPTION_OUT", "/kaggle/working"))
    args = parser.parse_args()
    bundle = find_bundle()
    run = json.loads((bundle / "run.json").read_text(encoding="utf-8"))
    out = Path(args.out)

    if args.model is None:  # one subprocess per model, so each gets a clean GPU
        summary = {}
        for model_path in run["models"]:
            r = subprocess.run([sys.executable, __file__, "--model", model_path, "--out", str(out)])
            report = out / model_path / "run_report.json"
            summary[model_path] = (json.loads(report.read_text(encoding="utf-8")) if report.is_file()
                                   else {"model_path": model_path, "failed": True, "exit_code": r.returncode})
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=1))
        return 0

    jobs = [json.loads(line) for line in (bundle / "jobs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    template = (bundle / run["prompt_file"]).read_text(encoding="utf-8")
    started = time.monotonic()
    try:
        backend = load_model(args.model, run["max_pixels"])
    except Exception as e:  # a path that will not install or fit is reported, not fatal to the others
        (out / args.model).mkdir(parents=True, exist_ok=True)
        (out / args.model / "load_error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
        print(f"[{args.model}] failed to load: {e}", flush=True)
        return 1
    print(f"[{args.model}] loaded {backend.model_id} in {time.monotonic() - started:.0f}s; GPU MB {gpu_memory_mb()}",
          flush=True)
    report = run_jobs(jobs, backend, bundle, out / args.model, args.model, run["prompt_version"], template)
    report["load_seconds"] = round(time.monotonic() - started - report["seconds_this_run"], 1)
    (out / args.model / "run_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
