from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from sqlalchemy.engine import make_url



ALLOWED_MODELS = frozenset({
    "gpt-5.6-sol",
    "gpt-4o-mini",
    "gpt-4.1-mini",
})


class Settings(BaseSettings):
    """
    Central application configuration.

    Values can be overridden through environment
    variables or the local .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =====================================================
    # Environment
    # =====================================================

    app_env: Literal[
        "test",
        "development",
        "production",
    ] = "development"


    # =====================================================
    # Authentication
    # =====================================================

    auth_cookie_name: str = "robbotgpt_session"

    auth_session_ttl_hours: int = Field(
        default=168,
        ge=1,
        le=720,
    )

    auth_cookie_samesite: Literal[
        "lax",
        "strict",
    ] = "lax"


    # =====================================================
    # OpenAI
    # =====================================================

    openai_model: str = "gpt-5.6-sol"

    embedding_model: str = (
        "text-embedding-3-small"
    )

    # =====================================================
    # API credentials
    # =====================================================

    openai_api_key: SecretStr | None = None

    tavily_api_key: SecretStr | None = None

    openweather_api_key: SecretStr | None = None

    alpha_vantage_api_key: SecretStr | None = None

    langsmith_api_key: SecretStr | None = None

    langsmith_endpoint: str = (
        "https://eu.api.smith.langchain.com"
    )

    langsmith_project: str = (
        "RobBotGPT"
    )

    langsmith_workspace_id: str | None = None


    # =====================================================
    # Storage
    # =====================================================

    data_dir: Path = Path("data")

    upload_dir: Path = Path("uploads")

    database_url: str

    redis_url: str = (
        "redis://localhost:6379/0"
    )


    # =====================================================
    # RAG
    # =====================================================

    rag_chunk_size: int = Field(
        default=900,
        ge=100,
        le=10_000,
    )

    rag_chunk_overlap: int = Field(
        default=150,
        ge=0,
    )

    rag_top_k: int = Field(
        default=4,
        ge=1,
        le=50,
    )

    rag_retrieval_top_k: int = Field(
        default=12,
        ge=1,
        le=50,
    )

    rag_final_top_k: int = Field(
        default=4,
        ge=1,
        le=50,
    )

    rag_relevance_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )

    rag_context_max_chars: int = Field(
        default=6000,
        ge=500,
        le=50_000,
    )

    # =====================================================
    # FastAPI / Uvicorn
    # =====================================================

    app_host: str = "0.0.0.0"

    app_port: int = Field(
        default=8080,
        ge=1,
        le=65_535,
    )

    app_reload: bool = True

    # =====================================================
    # Validators
    # =====================================================

    @field_validator(
        "openai_model"
    )
    @classmethod
    def validate_openai_model(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if value not in ALLOWED_MODELS:
            raise ValueError(
                f"Unsupported OPENAI_MODEL: {value}. "
                f"Allowed models: "
                f"{', '.join(sorted(ALLOWED_MODELS))}"
            )

        return value

    @field_validator(
        "embedding_model",
        "database_url",
        "app_host",
        "auth_cookie_name",
        "redis_url",
    )
    @classmethod
    def validate_non_empty_string(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Configuration value "
                "cannot be empty."
            )

        return value

    @model_validator(
        mode="after"
    )
    def validate_configuration(
        self,
    ):
        if (
            self.rag_chunk_overlap
            >= self.rag_chunk_size
        ):
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be "
                "smaller than RAG_CHUNK_SIZE."
            )

        if (
            self.rag_final_top_k
            > self.rag_retrieval_top_k
        ):
            raise ValueError(
                "RAG_FINAL_TOP_K cannot be greater "
                "than RAG_RETRIEVAL_TOP_K."
            )


        if (
            self.app_env == "production"
            and self.app_reload
        ):
            raise ValueError(
                "APP_RELOAD must be false "
                "in production."
            )

        database_backend = (
            make_url(
                self.database_url
            ).get_backend_name()
        )

        if (
            self.app_env
            in {
                "development",
                "production",
            }
            and database_backend
            != "postgresql"
        ):
            raise ValueError(
                "DATABASE_URL must use PostgreSQL "
                "in development and production."
            )

        return self

    # =====================================================
    # Environment helpers
    # =====================================================

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @property
    def is_development(self) -> bool:
        return (
            self.app_env
            == "development"
        )

    @property
    def is_production(self) -> bool:
        return (
            self.app_env
            == "production"
        )

    @property
    def auth_cookie_secure(self) -> bool:
        return self.is_production

    # =========================================================
    # Logging
    # =========================================================

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"



settings = Settings()