"""Central, environment-driven configuration for the RAG chatbot.

This module is the single source of truth for every tunable value (API keys,
model names, storage paths, retrieval parameters). It is imported by the
ingestion pipeline, the backend services, and the NIM smoke test, so nothing is
hard-coded anywhere else.

Design rules (see Implementation plan/TASK_LIST.md, Task 0.3):
  * Values come from a local ``.env`` file (loaded via python-dotenv) and fall
    back to sane defaults, so the config always loads - even with no ``.env``.
  * Storage paths are resolved to absolute paths against the project root, so
    behaviour does not depend on the current working directory.
  * The NVIDIA API key is OPTIONAL at import time: ingestion (local embeddings)
    must run with no credentials. Code that actually calls the LLM should invoke
    :func:`require_nvidia_api_key` to fail fast with an actionable message.

Usage::

    from backend.config import settings
    print(settings)                       # human-readable, secret redacted
    key = settings.require_nvidia_api_key()   # raises if missing
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

# Project root = the directory that contains this ``backend`` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load ``.env`` from the project root (no-op if the file is absent). ``override``
# is False so real environment variables take precedence over the .env file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


# --- Defaults (mirror .env.example) ------------------------------------------
_DEFAULTS: dict[str, str] = {
    "NVIDIA_API_KEY": "",
    "NVIDIA_MODEL": "meta/llama-3.3-70b-instruct",
    "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
    "LLM_TEMPERATURE": "0.1",
    "LLM_MAX_TOKENS": "1024",
    "EMBED_MODEL": "BAAI/bge-small-en-v1.5",
    "PDF_DIR": "pdf_for_RAG",
    "CHROMA_DIR": "data/chroma",
    "ASSETS_DIR": "data/assets",
    "METADATA_PATH": "data/metadata.json",
    "SQLITE_PATH": "data/app.db",
    "COLLECTION_NAME": "annual_reports",
    "CORS_ORIGINS": "http://localhost:5173,http://127.0.0.1:5173",
    "TOP_K": "8",
    "VISUAL_K": "3",
    "VISUAL_SIM_THRESHOLD": "0.3",
    # Reranking stage (retrieval -> rerank -> LLM)
    "RERANK_ENABLED": "true",
    "RERANK_BACKEND": "nvidia",                    # nvidia (hosted NIM) | local
    "RERANK_MODEL": "nv-rerank-qa-mistral-4b:1",
    "RERANK_BASE_URL": "",                         # blank -> SDK hosted default; set for a self-hosted NIM
    "RERANK_CANDIDATES": "24",                     # candidates pulled from Chroma before reranking
    "RERANK_TOP_N": "5",                           # reranked chunks handed to the LLM
}


def _env(key: str) -> str:
    """Return an env var, falling back to the documented default.

    Empty strings fall back to the default too, so a blank line in ``.env``
    (e.g. ``TOP_K=``) does not break type coercion.
    """
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return _DEFAULTS[key]
    return value.strip()


def _as_bool(value: str) -> bool:
    """Parse a truthy string (``true``/``1``/``yes``/``on``) into a bool."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(raw: str) -> Path:
    """Resolve a (possibly relative) path against the project root, absolutely."""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class Settings(BaseModel):
    """Immutable, validated application settings loaded from the environment."""

    model_config = ConfigDict(frozen=True)

    # Project root (computed, not from env) - handy for resolving other paths.
    project_root: Path = PROJECT_ROOT

    # NVIDIA NIM (generation LLM)
    NVIDIA_API_KEY: SecretStr = SecretStr("")
    NVIDIA_MODEL: str = _DEFAULTS["NVIDIA_MODEL"]
    NVIDIA_BASE_URL: str = _DEFAULTS["NVIDIA_BASE_URL"]
    LLM_TEMPERATURE: float = float(_DEFAULTS["LLM_TEMPERATURE"])
    LLM_MAX_TOKENS: int = int(_DEFAULTS["LLM_MAX_TOKENS"])

    # Embeddings (local)
    EMBED_MODEL: str = _DEFAULTS["EMBED_MODEL"]

    # Storage paths (absolute after validation)
    PDF_DIR: Path = _resolve_path(_DEFAULTS["PDF_DIR"])
    CHROMA_DIR: Path = _resolve_path(_DEFAULTS["CHROMA_DIR"])
    ASSETS_DIR: Path = _resolve_path(_DEFAULTS["ASSETS_DIR"])
    METADATA_PATH: Path = _resolve_path(_DEFAULTS["METADATA_PATH"])
    SQLITE_PATH: Path = _resolve_path(_DEFAULTS["SQLITE_PATH"])

    # Vector store / retrieval tuning
    COLLECTION_NAME: str = _DEFAULTS["COLLECTION_NAME"]
    CORS_ORIGINS: str = _DEFAULTS["CORS_ORIGINS"]  # comma-separated
    TOP_K: int = int(_DEFAULTS["TOP_K"])
    VISUAL_K: int = int(_DEFAULTS["VISUAL_K"])
    VISUAL_SIM_THRESHOLD: float = float(_DEFAULTS["VISUAL_SIM_THRESHOLD"])

    # Reranking (retrieval -> rerank -> LLM). When disabled, retrieval falls back
    # to the plain top-``TOP_K`` path, so behaviour is byte-for-byte the legacy one.
    RERANK_ENABLED: bool = _as_bool(_DEFAULTS["RERANK_ENABLED"])
    RERANK_BACKEND: str = _DEFAULTS["RERANK_BACKEND"]
    RERANK_MODEL: str = _DEFAULTS["RERANK_MODEL"]
    RERANK_BASE_URL: str = _DEFAULTS["RERANK_BASE_URL"]
    RERANK_CANDIDATES: int = int(_DEFAULTS["RERANK_CANDIDATES"])
    RERANK_TOP_N: int = int(_DEFAULTS["RERANK_TOP_N"])

    @field_validator(
        "PDF_DIR", "CHROMA_DIR", "ASSETS_DIR", "METADATA_PATH", "SQLITE_PATH",
        mode="before",
    )
    @classmethod
    def _abs_path(cls, value: object) -> Path:
        return _resolve_path(str(value))

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS_ORIGINS parsed into a list of allowed origins."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # --- Credential helpers --------------------------------------------------
    @property
    def has_nvidia_key(self) -> bool:
        """True if an NVIDIA API key is configured."""
        return bool(self.NVIDIA_API_KEY.get_secret_value())

    def require_nvidia_api_key(self) -> str:
        """Return the NVIDIA API key or raise a clear, actionable error.

        Call this from LLM code paths (smoke test, ``backend/llm.py``) - NOT at
        import time - so ingestion can run without credentials.
        """
        key = self.NVIDIA_API_KEY.get_secret_value()
        if not key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Get a free key at "
                "https://build.nvidia.com, then add it to your .env file:\n"
                "    NVIDIA_API_KEY=nvapi-xxxxxxxx\n"
                f"(.env is read from {self.project_root / '.env'})"
            )
        return key

    # --- Filesystem helpers --------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create storage directories if they do not exist (call from pipelines)."""
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        self.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Readable, secret-safe representation --------------------------------
    def __str__(self) -> str:
        key_state = "set" if self.has_nvidia_key else "MISSING"
        return (
            "Settings(\n"
            f"  project_root         = {self.project_root}\n"
            f"  NVIDIA_API_KEY       = <{key_state}>\n"
            f"  NVIDIA_MODEL         = {self.NVIDIA_MODEL}\n"
            f"  NVIDIA_BASE_URL      = {self.NVIDIA_BASE_URL}\n"
            f"  LLM_TEMPERATURE      = {self.LLM_TEMPERATURE}\n"
            f"  LLM_MAX_TOKENS       = {self.LLM_MAX_TOKENS}\n"
            f"  EMBED_MODEL          = {self.EMBED_MODEL}\n"
            f"  PDF_DIR              = {self.PDF_DIR}\n"
            f"  CHROMA_DIR           = {self.CHROMA_DIR}\n"
            f"  ASSETS_DIR           = {self.ASSETS_DIR}\n"
            f"  METADATA_PATH        = {self.METADATA_PATH}\n"
            f"  SQLITE_PATH          = {self.SQLITE_PATH}\n"
            f"  COLLECTION_NAME      = {self.COLLECTION_NAME}\n"
            f"  CORS_ORIGINS         = {self.CORS_ORIGINS}\n"
            f"  TOP_K                = {self.TOP_K}\n"
            f"  VISUAL_K             = {self.VISUAL_K}\n"
            f"  VISUAL_SIM_THRESHOLD = {self.VISUAL_SIM_THRESHOLD}\n"
            f"  RERANK_ENABLED       = {self.RERANK_ENABLED}\n"
            f"  RERANK_BACKEND       = {self.RERANK_BACKEND}\n"
            f"  RERANK_MODEL         = {self.RERANK_MODEL}\n"
            f"  RERANK_BASE_URL      = {self.RERANK_BASE_URL or '<hosted default>'}\n"
            f"  RERANK_CANDIDATES    = {self.RERANK_CANDIDATES}\n"
            f"  RERANK_TOP_N         = {self.RERANK_TOP_N}\n"
            ")"
        )


def _build_settings() -> Settings:
    """Construct the Settings singleton from the current environment."""
    return Settings(
        NVIDIA_API_KEY=SecretStr(_env("NVIDIA_API_KEY")),
        NVIDIA_MODEL=_env("NVIDIA_MODEL"),
        NVIDIA_BASE_URL=_env("NVIDIA_BASE_URL"),
        LLM_TEMPERATURE=float(_env("LLM_TEMPERATURE")),
        LLM_MAX_TOKENS=int(_env("LLM_MAX_TOKENS")),
        EMBED_MODEL=_env("EMBED_MODEL"),
        PDF_DIR=_env("PDF_DIR"),
        CHROMA_DIR=_env("CHROMA_DIR"),
        ASSETS_DIR=_env("ASSETS_DIR"),
        METADATA_PATH=_env("METADATA_PATH"),
        SQLITE_PATH=_env("SQLITE_PATH"),
        COLLECTION_NAME=_env("COLLECTION_NAME"),
        CORS_ORIGINS=_env("CORS_ORIGINS"),
        TOP_K=int(_env("TOP_K")),
        VISUAL_K=int(_env("VISUAL_K")),
        VISUAL_SIM_THRESHOLD=float(_env("VISUAL_SIM_THRESHOLD")),
        RERANK_ENABLED=_as_bool(_env("RERANK_ENABLED")),
        RERANK_BACKEND=_env("RERANK_BACKEND"),
        RERANK_MODEL=_env("RERANK_MODEL"),
        RERANK_BASE_URL=_env("RERANK_BASE_URL"),
        RERANK_CANDIDATES=int(_env("RERANK_CANDIDATES")),
        RERANK_TOP_N=int(_env("RERANK_TOP_N")),
    )


# Module-level singleton: ``from backend.config import settings``.
settings: Settings = _build_settings()


# Convenience module-level alias so callers can also do
# ``from backend.config import require_nvidia_api_key``.
def require_nvidia_api_key() -> str:
    """Module-level shortcut for :meth:`Settings.require_nvidia_api_key`."""
    return settings.require_nvidia_api_key()


if __name__ == "__main__":  # Manual check: ``python -m backend.config``
    print(settings)
