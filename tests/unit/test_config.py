"""Config validation rules (LLD §2)."""

from __future__ import annotations

import pytest

from mmrag.config import PROJECT_ROOT, ConfigError, load_settings
from tests.conftest import UNIT_ENV

pytestmark = pytest.mark.unit


def test_valid_config_loads(base_config, write_config):
    s = load_settings(write_config(base_config), UNIT_ENV)
    assert s.embed.dims == base_config["embed"]["dims"]
    assert s.secrets.database_url.get_secret_value() == UNIT_ENV["DATABASE_URL"]


def test_openai_key_is_optional(base_config, write_config):
    s = load_settings(write_config(base_config), {**UNIT_ENV, "OPENAI_API_KEY": ""})
    assert s.secrets.openai_api_key is None


def test_secrets_are_masked_in_repr(base_config, write_config):
    s = load_settings(write_config(base_config), {**UNIT_ENV, "OPENAI_API_KEY": "sk-secret-value"})
    assert "sk-secret-value" not in repr(s)
    assert s.secrets.openai_api_key.get_secret_value() == "sk-secret-value"


def test_unknown_key_rejected(base_config, write_config):
    base_config["parse"]["not_a_setting"] = 1
    with pytest.raises(ConfigError, match="not_a_setting"):
        load_settings(write_config(base_config), UNIT_ENV)


def test_embedding_dims_above_hnsw_limit_rejected(base_config, write_config):
    base_config["embed"]["dims"] = 3072
    with pytest.raises(ConfigError, match="embed.dims"):
        load_settings(write_config(base_config), UNIT_ENV)


def test_wrong_type_rejected(base_config, write_config):
    base_config["search"]["top_k"] = "eight"
    with pytest.raises(ConfigError, match="top_k"):
        load_settings(write_config(base_config), UNIT_ENV)


def test_secrets_section_in_yaml_rejected(base_config, write_config):
    base_config["secrets"] = {"openai_api_key": "sk-oops"}
    with pytest.raises(ConfigError, match="Secrets must not be in config.yaml"):
        load_settings(write_config(base_config), UNIT_ENV)


def test_missing_database_url_rejected(base_config, write_config):
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        load_settings(write_config(base_config), {})


def test_missing_config_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "missing.yaml", UNIT_ENV)


def test_relative_paths_resolve_to_project_root(base_config, write_config):
    s = load_settings(write_config(base_config), UNIT_ENV)
    assert s.resolve(s.paths.pdf_dir) == PROJECT_ROOT / "data" / "pdfs"


def test_project_config_is_valid():
    """The committed config.yaml itself must always validate."""
    load_settings(PROJECT_ROOT / "config.yaml", UNIT_ENV)


# ---------------------------------------------------------------- Phase 10: experiment configs

def test_gemma_config_extends_the_project_config():
    s = load_settings(PROJECT_ROOT / "config.gemma.yaml", UNIT_ENV)
    base = load_settings(PROJECT_ROOT / "config.yaml", UNIT_ENV)
    assert (s.embed.provider, s.embed.model, s.embed.dims) == ("local", "google/embeddinggemma-2", 768)
    assert s.agent == base.agent and s.search == base.search  # everything else is inherited
    assert s.secrets.database_url.get_secret_value() == "postgresql://user:pw@localhost:5433/mmrag_gemma"


def test_extends_merges_sections_key_by_key(tmp_path, base_config, write_config):
    write_config(base_config)  # tmp_path/config.yaml
    child = tmp_path / "child.yaml"
    child.write_text("extends: config.yaml\nembed:\n  dims: 256\n", encoding="utf-8")
    s = load_settings(child, UNIT_ENV)
    assert s.embed.dims == 256 and s.embed.model == base_config["embed"]["model"]


def test_extends_loop_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="loop"):
        load_settings(tmp_path / "a.yaml", UNIT_ENV)


def test_database_name_replaces_only_the_name(base_config, write_config):
    base_config["database_name"] = "mmrag_other"
    env = {"DATABASE_URL": "postgresql://user:pw@db.example:5433/mmrag?sslmode=require"}
    s = load_settings(write_config(base_config), env)
    assert s.secrets.database_url.get_secret_value() == "postgresql://user:pw@db.example:5433/mmrag_other?sslmode=require"


def test_unsafe_database_name_rejected(base_config, write_config):
    base_config["database_name"] = "x; DROP DATABASE mmrag"
    with pytest.raises(ConfigError):
        load_settings(write_config(base_config), UNIT_ENV)


def test_judge_embedding_does_not_follow_retrieval(base_config, write_config):
    base_config["embed"].update(provider="local", model="google/embeddinggemma-2", dims=768)
    s = load_settings(write_config(base_config), UNIT_ENV)
    assert (s.eval.judge_embed_model, s.eval.judge_embed_dims) == ("text-embedding-3-large", 1536)
