"""Typed settings from config.yaml + .env (LLD §2).

Unknown keys, wrong types or invalid values stop the program at startup.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Mapping

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paths(_Section):
    pdf_dir: Path
    data_dir: Path


class Parse(_Section):
    render_dpi: int = Field(gt=0)
    min_image_px: int = Field(gt=0)
    textless_chars: int = Field(ge=0)
    furniture_width_ratio: float = Field(gt=0, le=1)
    line_aspect_ratio: float = Field(gt=1)
    cluster_merge_distance_pt: float = Field(gt=0)
    min_cluster_shapes: int = Field(ge=1)


class Caption(_Section):
    model_path: Literal["awq-7b", "nf4-7b", "3b"]
    max_pixels: int = Field(gt=0)
    prompt_version: str


class Enrich(_Section):
    label_overlap_min: float = Field(ge=0, le=1)
    proximity_window_pt: float = Field(gt=0)


class Chunk(_Section):
    min_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    overlap: float = Field(ge=0, lt=1)


class Embed(_Section):
    model: str
    # pgvector's HNSW index supports at most 2,000 dimensions for `vector`.
    dims: int = Field(gt=0, le=2000)
    batch_size: int = Field(gt=0, le=2048)


class Search(_Section):
    candidates: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    top_k: int = Field(gt=0)
    hnsw_ef_search: int = Field(gt=0)
    rerank: bool


class Agent(_Section):
    model: str | None
    effort: str | None
    summary_model: str | None
    rounds_default: int = Field(ge=1)
    rounds_multi: int = Field(ge=1)


class UI(_Section):
    page_dpi: int = Field(gt=0)


class Auth(_Section):
    min_password_length: int = Field(ge=8)
    common_passwords_file: Path
    max_failed_attempts: int = Field(ge=1)
    lockout_minutes: int = Field(ge=1)
    ip_max_failed_attempts: int = Field(ge=1)
    session_idle_hours: float = Field(gt=0)
    session_absolute_days: float = Field(gt=0)
    daily_question_limit: int = Field(ge=1)
    daily_cost_limit_usd: float = Field(gt=0)
    limits_timezone: str
    trust_proxy_headers: bool


class Observability(_Section):
    enabled: bool
    environment: Literal["dev", "prod"]
    otlp_traces_endpoint: str | None
    otlp_metrics_endpoint: str | None
    capture_content: bool
    sample_ratio: float = Field(ge=0, le=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    log_file: Path | None


class Retention(_Section):
    query_log_days: int = Field(ge=1)
    auth_events_days: int = Field(ge=1)


class Eval(_Section):
    golden_set: Path
    judge_model: str | None
    regression_tolerance: float = Field(ge=0, le=1)


class Secrets(_Section):
    openai_api_key: SecretStr | None
    database_url: SecretStr


class Settings(_Section):
    paths: Paths
    parse: Parse
    caption: Caption
    enrich: Enrich
    chunk: Chunk
    embed: Embed
    search: Search
    agent: Agent
    ui: UI
    auth: Auth
    observability: Observability
    retention: Retention
    eval: Eval
    secrets: Secrets

    def resolve(self, path: Path) -> Path:
        """Resolve a config path relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


class ConfigError(RuntimeError):
    pass


def load_settings(config_file: Path, environ: Mapping[str, str]) -> Settings:
    """Build settings from a YAML file and an environment mapping (no global state)."""
    try:
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as e:
        raise ConfigError(f"Config file not found: {config_file}") from e

    if "secrets" in raw:
        raise ConfigError("Secrets must not be in config.yaml; put them in .env")
    database_url = environ.get("DATABASE_URL")
    if not database_url:
        raise ConfigError("DATABASE_URL is not set (see .env.example)")
    raw["secrets"] = {
        "openai_api_key": environ.get("OPENAI_API_KEY") or None,
        "database_url": database_url,
    }

    try:
        return Settings.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration in {config_file}:\n{e}") from e


@lru_cache(maxsize=1)
def get_settings(config_file: Path | None = None) -> Settings:
    """Process-wide settings: .env + config.yaml (or the file named by MMRAG_CONFIG)."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = config_file or Path(os.environ.get("MMRAG_CONFIG") or PROJECT_ROOT / "config.yaml")
    return load_settings(path, os.environ)
