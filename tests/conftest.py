"""Shared fixtures. Everything is offline: hashing embedder, temp storage, no API key."""

from __future__ import annotations

from pathlib import Path

import pytest

from sellerpolicy.config import PROJECT_ROOT, Settings
from sellerpolicy.embeddings import HashingEmbedder
from sellerpolicy.generator import Generator
from sellerpolicy.pipeline import SellerPolicyRAG

SAMPLE_DOCS = {
    "appeals.md": """# Appeals Policy

Intro text about appeals for sellers.

## Deactivation and Appeals

### Appeal timelines

A seller may submit an appeal within 17 days of the deactivation notice.

### Plan of Action

A strong POA has three parts: root cause, corrective actions, preventive measures.
""",
    "claims.md": """# A-to-z Guarantee Claims

## Response window

The seller has 72 hours to respond to an A-to-z claim notification.

## Eligibility

A buyer can file a claim when the item did not arrive within 3 days after the estimated delivery date.
""",
    "returns.txt": """# Returns

## SAFE-T claims

A seller may file a SAFE-T claim within 30 days of the refund date.

## Restocking fees

A restocking fee of up to 20% may be charged when the item was opened or used.
""",
}


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    for name, text in SAMPLE_DOCS.items():
        (data / name).write_text(text, encoding="utf-8")
    return data


@pytest.fixture
def test_settings(tmp_path: Path, sample_dir: Path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=None,
        embedding_provider="hashing",
        hashing_dim=256,
        data_dir=sample_dir,
        storage_dir=tmp_path / "storage",
        use_reranker=False,
        top_k=3,
        candidate_k=10,
    )


@pytest.fixture
def rag(test_settings: Settings) -> SellerPolicyRAG:
    instance = SellerPolicyRAG(
        settings=test_settings,
        embedder=HashingEmbedder(test_settings.hashing_dim),
        generator=Generator(test_settings),
    )
    instance.build_index()
    return instance


@pytest.fixture
def real_policy_dir() -> Path:
    return PROJECT_ROOT / "data" / "policies"
