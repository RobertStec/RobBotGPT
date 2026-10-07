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


ALLOWED_MODELS = frozenset({
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
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
    # OpenAI
    # =====================================================

    openai_model: str = "gpt-4o-mini"

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


    # =====================================================
    # Storage
    # =====================================================

    data_dir: Path = Path("data")

    upload_dir: Path = Path("uploads")

    chroma_dir: Path = Path("chroma_db")

    database_url: str = (
        "sqlite:///data/chatbot_memory.db"
    )

    checkpoint_db_path: Path = Path(
        "data/langgraph_checkpoints.sqlite"
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
            self.app_env == "production"
            and self.app_reload
        ):
            raise ValueError(
                "APP_RELOAD must be false "
                "in production."
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