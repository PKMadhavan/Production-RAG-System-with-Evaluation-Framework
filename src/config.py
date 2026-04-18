"""Application configuration via environment variables."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "rag-api"
    app_env: str = "development"
    log_level: str = "INFO"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "raguser"
    postgres_password: str = "changeme"
    postgres_db: str = "ragdb"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl_query: int = 3600
    redis_ttl_embedding: int = 86400

    # Embeddings
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # File upload
    max_file_size_mb: int = 50

    # OpenAI (optional — used for RAGAS evaluation answer generation)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Evaluation
    eval_max_samples: int = 100

    # Hybrid retrieval
    bm25_k_param: int = 60  # RRF constant k

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024
