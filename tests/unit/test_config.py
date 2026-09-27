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
