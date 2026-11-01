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
        "REDIS_URL",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_CONTEXT_MAX_CHARS",
        "RAG_TOP_K",
        "APP_HOST",
        "APP_PORT",
        "APP_ENV",
        "APP_RELOAD",
        "AUTH_COOKIE_NAME",
        "AUTH_SESSION_TTL_HOURS",
        "AUTH_COOKIE_SAMESITE",
        "ALLOWED_HOSTS",
        "CORS_ALLOWED_ORIGINS",
        "OPENAI_API_KEY",
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
        == "gpt-5.6-sol"
    )

    assert (
        settings.embedding_model
        == "text-embedding-3-small"
    )

    assert settings.allowed_hosts == [
        "localhost",
        "127.0.0.1",
        "testserver",
        "robbotgpt.aispectra.pl",
    ]

    assert (
        settings.cors_allowed_origins
        == [
            "https://robbotgpt.aispectra.pl",
        ]
    )

    assert (
        settings.database_url
        == (
            "postgresql+psycopg://"
            "user:password@localhost:5433/test_db"
        )
    )

    assert (
        settings.redis_url
        == "redis://localhost:6379/0"
    )

    assert (
        settings.data_dir
        == Path("data")
    )

    assert (
        settings.upload_dir
        == Path("uploads")
    )

    assert (
        settings.rag_context_max_chars
        == 6000
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


    assert (
        settings.auth_cookie_name
        == "robbotgpt_session"
    )

    assert (
        settings.auth_session_ttl_hours
        == 168
    )

    assert (
        settings.auth_cookie_samesite
        == "lax"
    )

    assert settings.auth_cookie_secure is False




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

    monkeypatch.setenv(
        "REDIS_URL",
        "redis://test-redis:6379/1",
    )

    monkeypatch.setenv(
        "RAG_CONTEXT_MAX_CHARS",
        "8000",
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

    assert (
        settings.rag_context_max_chars
        == 8000
    )

    assert (
        settings.redis_url
        == "redis://test-redis:6379/1"
    )



def test_settings_reads_auth_environment_variables(
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTH_COOKIE_NAME",
        "custom_session",
    )

    monkeypatch.setenv(
        "AUTH_SESSION_TTL_HOURS",
        "24",
    )

    monkeypatch.setenv(
        "AUTH_COOKIE_SAMESITE",
        "strict",
    )

    settings = Settings(
        _env_file=None
    )

    assert (
        settings.auth_cookie_name
        == "custom_session"
    )

    assert (
        settings.auth_session_ttl_hours
        == 24
    )

    assert (
        settings.auth_cookie_samesite
        == "strict"
    )




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

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-openai-key",
    )

    settings = Settings(
        _env_file=None,
        database_url=(
            "postgresql+psycopg://"
            "user:password@localhost:5432/test_db"
        ),
    )

    assert (
        settings.app_env
        == "production"
    )

    assert settings.app_reload is False

    assert settings.is_production is True

    assert settings.auth_cookie_secure is True



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

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-openai-key",
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
        kwargs["openai_api_key"] = (
            "test-openai-key"
        )

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



@pytest.mark.parametrize(
    "ttl",
    [
        "0",
        "721",
    ],
)
def test_settings_rejects_invalid_auth_session_ttl(
    monkeypatch,
    ttl,
):
    monkeypatch.setenv(
        "AUTH_SESSION_TTL_HOURS",
        ttl,
    )

    with pytest.raises(
        ValidationError
    ):
        Settings(
            _env_file=None
        )




def test_upload_max_bytes_default():
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url="sqlite:///:memory:",
    )

    assert (
        settings.upload_max_bytes
        == 10 * 1024 * 1024
    )




def test_upload_max_bytes_can_be_set_from_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "UPLOAD_MAX_BYTES",
        "5242880",
    )

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url="sqlite:///:memory:",
    )

    assert (
        settings.upload_max_bytes
        == 5 * 1024 * 1024
    )



def test_settings_rejects_wildcard_allowed_host(
    monkeypatch,
):
    monkeypatch.setenv(
        "ALLOWED_HOSTS",
        '["*"]',
    )

    with pytest.raises(
        ValidationError,
        match="Wildcard host is not allowed",
    ):
        Settings(
            _env_file=None,
            app_env="test",
            database_url="sqlite:///:memory:",
        )



def test_settings_rejects_wildcard_cors_origin(
    monkeypatch,
):
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        '["*"]',
    )

    with pytest.raises(
        ValidationError,
        match=(
            "Wildcard CORS origin "
            "is not allowed"
        ),
    ):
        Settings(
            _env_file=None,
            app_env="test",
            database_url="sqlite:///:memory:",
        )



def test_settings_requires_openai_key_in_production(
    monkeypatch,
):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        ValidationError,
        match=(
            "OPENAI_API_KEY must be configured "
            "in production"
        ),
    ):
        Settings(
            _env_file=None,
            app_env="production",
            app_reload=False,
            database_url=(
                "postgresql+psycopg://"
                "user:password@localhost:5432/db"
            ),
        )




def test_settings_rejects_blank_openai_key_in_production(
    monkeypatch,
):
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "   ",
    )

    with pytest.raises(
        ValidationError,
        match=(
            "OPENAI_API_KEY must be configured "
            "in production"
        ),
    ):
        Settings(
            _env_file=None,
            app_env="production",
            app_reload=False,
            database_url=(
                "postgresql+psycopg://"
                "user:password@localhost:5432/db"
            ),
        )




def test_database_and_redis_urls_are_hidden_from_repr():
    database_url = (
        "postgresql+psycopg://"
        "user:database-secret@localhost:5432/db"
    )

    redis_url = (
        "redis://:redis-secret@localhost:6379/0"
    )

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=database_url,
        redis_url=redis_url,
    )

    settings_repr = repr(
        settings
    )

    assert "database-secret" not in settings_repr
    assert "redis-secret" not in settings_repr




def test_production_configuration_can_be_loaded_from_environment(
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

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-openai-key",
    )

    monkeypatch.setenv(
        "OPENAI_MODEL",
        "gpt-5.6-sol",
    )

    monkeypatch.setenv(
        "DATABASE_URL",
        (
            "postgresql+psycopg://"
            "user:password@db:5432/robbotgpt"
        ),
    )

    monkeypatch.setenv(
        "REDIS_URL",
        "redis://redis:6379/0",
    )

    settings = Settings(
        _env_file=None
    )

    assert settings.is_production is True
    assert settings.app_reload is False

    assert (
        settings.openai_model
        == "gpt-5.6-sol"
    )

    assert (
        settings.openai_api_key
        .get_secret_value()
        == "test-openai-key"
    )

    assert (
        settings.database_url
        == (
            "postgresql+psycopg://"
            "user:password@db:5432/robbotgpt"
        )
    )

    assert (
        settings.redis_url
        == "redis://redis:6379/0"
    )




def test_optional_api_keys_are_not_required_in_production(
    monkeypatch,
):
    optional_keys = [
        "TAVILY_API_KEY",
        "OPENWEATHER_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "LANGSMITH_API_KEY",
    ]

    for key in optional_keys:
        monkeypatch.delenv(
            key,
            raising=False,
        )

    settings = Settings(
        _env_file=None,
        app_env="production",
        app_reload=False,
        openai_api_key="test-openai-key",
        database_url=(
            "postgresql+psycopg://"
            "user:password@db:5432/robbotgpt"
        ),
    )

    assert settings.tavily_api_key is None
    assert settings.openweather_api_key is None
    assert settings.alpha_vantage_api_key is None
    assert settings.langsmith_api_key is None