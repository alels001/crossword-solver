"""Central configuration, loaded from environment variables or a ``.env`` file.

Every knob that changes behaviour in production (model names, budgets, timeouts,
feature flags) lives here so deployments can be tuned without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

Provider = Literal["openai", "anthropic", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CWS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ---------------------------------------------------------------
    llm_provider: Provider = Field(
        default="fake",
        description="Which LLM backend to use. 'fake' runs fully offline (tests, demos without a key).",
    )
    openai_model: str = "gpt-4.1-mini"
    openai_vision_model: str = "gpt-4.1"
    openai_embedding_model: str = "text-embedding-3-small"
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_vision_model: str = "claude-sonnet-5-5"
    temperature: float = 0.2
    max_output_tokens: int = 800
    request_timeout_s: float = 60.0
    max_retries: int = 3

    # ---- Caching / budgets -------------------------------------------------
    cache_dir: Path = Path(".cache")
    enable_llm_cache: bool = True
    max_cost_usd_per_puzzle: float = Field(
        default=0.50, description="Hard budget: LLM calls stop once a single solve exceeds this."
    )

    # ---- Retrieval (RAG) ---------------------------------------------------
    knowledge_base_path: Path = REPO_ROOT / "data" / "knowledge" / "clues_de.csv"
    benchmark_dir: Path = REPO_ROOT / "data" / "benchmark"
    retriever: Literal["tfidf", "embedding", "hybrid"] = "tfidf"
    rag_top_k: int = 8

    # ---- Candidate sources -------------------------------------------------
    enable_web_dictionary: bool = Field(
        default=False, description="Scrape wort-suchen.de (polite, cached). Off by default."
    )
    web_dictionary_delay_s: float = 0.5
    max_candidates_per_entry: int = 40

    # ---- Solver / agent ----------------------------------------------------
    csp_max_nodes: int = 60_000
    agent_max_steps: int = 25

    # ---- Vision ------------------------------------------------------------
    grid_rows: int = 13
    grid_cols: int = 10
    max_image_side: int = 1600
    # Rows/cols (inclusive) covered by the printed solution box; excluded from the grid.
    mask_region: tuple[int, int, int, int] | None = (8, 6, 12, 9)

    # ---- API ---------------------------------------------------------------
    api_key: str | None = Field(default=None, description="If set, the HTTP API requires X-API-Key.")
    max_upload_mb: int = 10
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
