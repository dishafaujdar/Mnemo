"""Pydantic settings loaded from environment."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Auth
    mnemo_api_key: str = ""

    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    extraction_model: str = "gpt-4o-mini"

    # Storage
    database_url: str = "sqlite+aiosqlite:///./mnemo.db"
    qdrant_path: str = "./qdrant_storage"
    redis_url: str = "redis://localhost:6379/0"

    # Extraction
    spacy_model: str = "en_core_web_sm"
    llm_extraction_threshold: float = 0.6
    extraction_concurrency: int = 5

    # Retrieval
    default_token_budget: int = 1500
    default_search_limit: int = 20
    bm25_weight: float = 0.4
    vector_weight: float = 0.6


settings = Settings()
