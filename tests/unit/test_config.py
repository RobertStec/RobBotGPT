from pathlib import Path

import pytest

from pydantic import ValidationError, SecretStr

from core.config import (
    ALLOWED_MODELS,
    Settings,
)


def test_settings_default_values(
    monkeypatch,
):
    environment_variables = [
        "OPENAI_MODEL",
        "EMBEDDING_MODEL",
        "DATA_DIR",
        "UPLOAD_DIR",
        "DATABASE_URL",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_TOP_K",
        "APP_HOST",
        "APP_PORT",
        "APP_ENV",
        "APP_RELOAD",
    ]

    for variable in environment_variables:
        monkeypatch.delenv(
            variable,
            raising=False,
        )

    settings = Settings(
        _env_file=None,
        database_url=(
            "postgresql+psycopg://"
            "user:password@localhost:5433/test_db"
        ),
    )

    assert (
        settings.openai_model
        == "gpt-4o-mini"
    )

    assert (
        settings.embedding_model
        == "text-embedding-3-small"
    )

    assert (
        settings.database_url
        == (
            "postgresql+psycopg://"
            "user:password@localhost:5433/test_db"
        )
    )

    assert (
        settings.data_dir
        == Path("data")
    )

    assert (
        settings.upload_dir
        == Path("uploads")
    )


    assert settings.rag_chunk_size == 900
    assert settings.rag_chunk_overlap == 150
    assert settings.rag_top_k == 4

    assert settings.app_port == 8080


    assert (
    settings.app_env
    == "development"
    )

    assert settings.app_reload is True

    assert settings.is_development is True
    assert settings.is_test is False
    assert settings.is_production is False




def test_settings_reads_environment_variables(
    monkeypatch,
):
    monkeypatch.setenv(
        "OPENAI_MODEL",
        "gpt-4.1-mini",
    )

    monkeypatch.setenv(
        "RAG_TOP_K",
        "8",
    )

    monkeypatch.setenv(
        "APP_PORT",
        "9000",
    )

    settings = Settings(
        _env_file=None
    )

    assert (
        settings.openai_model
        == "gpt-4.1-mini"
    )

    assert settings.rag_top_k == 8
    assert settings.app_port == 9000


def test_allowed_models_contains_supported_models():
    assert "gpt-4o-mini" in ALLOWED_MODELS
    assert "gpt-4.1-mini" in ALLOWED_MODELS



def test_settings_supports_test_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "APP_ENV",
        "test",
    )

    settings = Settings(
        _env_file=None
    )

    assert settings.app_env == "test"

    assert settings.is_test is True
    assert settings.is_development is False
    assert settings.is_production is False



def test_settings_supports_production_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "APP_ENV",
        "production",
    )

    monkeypatch.setenv(
        "APP_RELOAD",
        "false",
    )

    settings = Settings(
        _env_file=None
    )

    assert (
        settings.app_env
        == "production"
    )

    assert settings.app_reload is False

    assert settings.is_production is True



def test_settings_rejects_reload_in_production(
    monkeypatch,
):
    monkeypatch.setenv(
        "APP_ENV",
        "production",
    )

    monkeypatch.setenv(
        "APP_RELOAD",
        "true",
    )

    with pytest.raises(
        ValidationError,
        match=(
            "APP_RELOAD must be false "
            "in production"
        ),
    ):
        Settings(
            _env_file=None
        )



def test_settings_rejects_unsupported_openai_model(
    monkeypatch,
):
    monkeypatch.setenv(
        "OPENAI_MODEL",
        "invalid-model",
    )

    with pytest.raises(
        ValidationError,
        match="Unsupported OPENAI_MODEL",
    ):
        Settings(
            _env_file=None
        )




@pytest.mark.parametrize(
    (
        "variable",
        "value",
    ),
    [
        (
            "RAG_CHUNK_SIZE",
            "50",
        ),
        (
            "RAG_TOP_K",
            "0",
        ),
        (
            "RAG_TOP_K",
            "51",
        ),
    ],
)
def test_settings_rejects_invalid_rag_limits(
    monkeypatch,
    variable,
    value,
):
    monkeypatch.setenv(
        variable,
        value,
    )

    with pytest.raises(
        ValidationError
    ):
        Settings(
            _env_file=None
        )



def test_settings_rejects_invalid_chunk_overlap(
    monkeypatch,
):
    monkeypatch.setenv(
        "RAG_CHUNK_SIZE",
        "500",
    )

    monkeypatch.setenv(
        "RAG_CHUNK_OVERLAP",
        "500",
    )

    with pytest.raises(
        ValidationError,
        match=(
            "RAG_CHUNK_OVERLAP must be "
            "smaller than RAG_CHUNK_SIZE"
        ),
    ):
        Settings(
            _env_file=None
        )



@pytest.mark.parametrize(
    "port",
    [
        "0",
        "65536",
    ],
)
def test_settings_rejects_invalid_port(
    monkeypatch,
    port,
):
    monkeypatch.setenv(
        "APP_PORT",
        port,
    )

    with pytest.raises(
        ValidationError
    ):
        Settings(
            _env_file=None
        )



def test_settings_parses_environment_types(
    monkeypatch,
):
    monkeypatch.setenv(
        "RAG_TOP_K",
        "10",
    )

    monkeypatch.setenv(
        "APP_PORT",
        "9000",
    )

    monkeypatch.setenv(
        "APP_RELOAD",
        "false",
    )

    settings = Settings(
        _env_file=None
    )

    assert settings.rag_top_k == 10
    assert isinstance(
        settings.rag_top_k,
        int,
    )

    assert settings.app_port == 9000
    assert isinstance(
        settings.app_port,
        int,
    )

    assert settings.app_reload is False
    assert isinstance(
        settings.app_reload,
        bool,
    )



# Test sekretów

def test_settings_load_api_credentials_as_secrets(
    monkeypatch,
):
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "openai-test-key",
    )

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "tavily-test-key",
    )

    settings = Settings(
        _env_file=None
    )

    assert isinstance(
        settings.openai_api_key,
        SecretStr,
    )

    assert (
        settings.openai_api_key
        .get_secret_value()
        == "openai-test-key"
    )

    assert isinstance(
        settings.tavily_api_key,
        SecretStr,
    )

    assert (
        settings.tavily_api_key
        .get_secret_value()
        == "tavily-test-key"
    )




# Test bezpieczeństwa

def test_secret_values_are_not_exposed_in_repr(
    monkeypatch,
):
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "super-secret-openai-key",
    )

    settings = Settings(
        _env_file=None
    )

    assert (
        "super-secret-openai-key"
        not in repr(settings)
    )




def test_settings_allows_sqlite_in_test_environment():
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url="sqlite:///:memory:",
    )

    assert (
        settings.database_url
        == "sqlite:///:memory:"
    )




@pytest.mark.parametrize(
    "app_env",
    [
        "development",
        "production",
    ],
)
def test_settings_rejects_sqlite_outside_test_environment(
    app_env,
):
    kwargs = {
        "_env_file": None,
        "app_env": app_env,
        "database_url": "sqlite:///app.db",
    }

    if app_env == "production":
        kwargs["app_reload"] = False

    with pytest.raises(
        ValueError,
        match="must use PostgreSQL",
    ):
        Settings(**kwargs)




def test_settings_allows_postgresql_in_development():
    settings = Settings(
        _env_file=None,
        app_env="development",
        database_url=(
            "postgresql+psycopg://"
            "user:password@localhost:5433/db"
        ),
    )

    assert (
        settings.database_url.startswith(
            "postgresql+psycopg://"
        )
    )