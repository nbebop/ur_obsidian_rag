"""Load and validate configuration from `config.yaml` + `.env`.

This is the single source of truth for the vault path and model choices.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Project root = two levels up from this file (src/obsidian_rag/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

TOKEN_ENV_VARS = ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN")


@dataclass
class Config:
    vault_path: Path
    embedding_model: str
    llm_model: str
    llm_provider: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    storage_dir: Path
    collection_name: str
    hf_token: str

    @property
    def chroma_path(self) -> Path:
        return self.storage_dir / "chroma"


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(
    config_path: os.PathLike | str | None = None,
    *,
    require_vault: bool = True,
    require_token: bool = True,
) -> Config:
    """Read config.yaml and .env, validate, and return a Config.

    `require_vault` / `require_token` let callers (e.g. tests, indexing vs.
    querying) relax validation when a given resource isn't needed yet.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy the provided config.yaml and edit it."
        )

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    missing = [k for k in ("vault_path", "embedding_model", "llm_model") if k not in raw]
    if missing:
        raise ValueError(f"config.yaml is missing required keys: {', '.join(missing)}")

    vault_path = _resolve(str(raw["vault_path"]))
    if require_vault and not vault_path.exists():
        raise FileNotFoundError(
            f"vault_path does not exist: {vault_path}. Edit `vault_path` in config.yaml."
        )

    hf_token = ""
    for var in TOKEN_ENV_VARS:
        if os.getenv(var):
            hf_token = os.environ[var]
            break
    if require_token and not hf_token:
        raise ValueError(
            "No HuggingFace token found. Copy .env.example to .env and set "
            "HUGGINGFACEHUB_API_TOKEN (create one at "
            "https://huggingface.co/settings/tokens)."
        )

    return Config(
        vault_path=vault_path,
        embedding_model=str(raw["embedding_model"]),
        llm_model=str(raw["llm_model"]),
        llm_provider=str(raw.get("llm_provider", "auto")),
        chunk_size=int(raw.get("chunk_size", 512)),
        chunk_overlap=int(raw.get("chunk_overlap", 64)),
        top_k=int(raw.get("top_k", 5)),
        storage_dir=_resolve(str(raw.get("storage_dir", "./storage"))),
        collection_name=str(raw.get("collection_name", "obsidian")),
        hf_token=hf_token,
    )
