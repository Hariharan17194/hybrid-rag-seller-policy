"""All settings in one place.

Seller Support analogy: this is the team's "desk setup" sheet. Everyone works from
the same values (which model, how many sources to quote), and a single associate can
override one line in their own `.env` without editing the shared sheet.

Priority (highest wins): environment variable > `.env` file > default below.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_PROVIDERS = ("sentence-transformers", "openai", "hashing")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Embeddings
    embedding_provider: str = "sentence-transformers"
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    hashing_dim: int = 1024

    # Retrieval
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"

    # Chunking
    chunk_max_chars: int = 1000
    chunk_overlap_chars: int = 150

    # Paths
    data_dir: Path = PROJECT_ROOT / "data" / "policies"
    storage_dir: Path = PROJECT_ROOT / "storage"
    collection_name: str = "seller_policies"

    @field_validator("embedding_provider")
    @classmethod
    def _check_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in EMBEDDING_PROVIDERS:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {EMBEDDING_PROVIDERS}, got {value!r}")
        return value

    @field_validator("data_dir", "storage_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())


settings = Settings()
