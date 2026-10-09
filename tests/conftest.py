import os
import subprocess
import importlib
import sys

import pytest
import database

from types import ModuleType

from fastapi.testclient import TestClient

from sqlalchemy import text
from sqlalchemy.engine import make_url

from core.environment import (
    configure_environment,
)


configure_environment()



@pytest.fixture
def test_database(
    tmp_path,
    monkeypatch,
):
    db_path = (
        tmp_path
        / "test_chatbot.db"
    )

    test_database_url = (
        f"sqlite:///{db_path}"
    )

    database.clear_database_resource_cache()

    monkeypatch.setattr(
        database.settings,
        "app_env",
        "test",
    )

    monkeypatch.setattr(
        database.settings,
        "database_url",
        test_database_url,
    )

    test_engine = (
        database.get_engine()
    )

    TestSessionLocal = (
        database.get_session_factory()
    )

    database.Base.metadata.create_all(
        bind=test_engine
    )

    yield TestSessionLocal

    database.Base.metadata.drop_all(
        bind=test_engine
    )

    database.clear_database_resource_cache()




@pytest.fixture(scope="session")
def postgres_test_database_url():
    database_url = os.getenv(
        "TEST_DATABASE_URL"
    )

    if not database_url:
        pytest.skip(
            "TEST_DATABASE_URL is not configured."
        )

    url = make_url(
        database_url
    )

    if (
        url.get_backend_name()
        != "postgresql"
    ):
        raise RuntimeError(
            "TEST_DATABASE_URL must use PostgreSQL."
        )

    if not (
        url.database
        and url.database.endswith(
            "_test"
        )
    ):
        raise RuntimeError(
            "Refusing to run PostgreSQL integration "
            "tests against a non-test database."
        )

    return database_url




@pytest.fixture(scope="session")
def postgres_schema(
    postgres_test_database_url,
):
    environment = os.environ.copy()

    environment[
        "DATABASE_URL"
    ] = postgres_test_database_url

    subprocess.run(
        [
            "alembic",
            "upgrade",
            "head",
        ],
        check=True,
        env=environment,
    )

    return postgres_test_database_url




@pytest.fixture
def postgres_database(
    postgres_schema,
    monkeypatch,
):
    database.clear_database_resource_cache()

    monkeypatch.setattr(
        database.settings,
        "database_url",
        postgres_schema,
    )

    engine = database.get_engine()

    session_factory = (
        database.get_session_factory()
    )

    tables = (
        "rag_chunks",
        "rag_documents",
        "chat_messages",
        "long_term_memory",
        "conversations",
    )

    with engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE TABLE "
                + ", ".join(tables)
                + " RESTART IDENTITY CASCADE"
            )
        )

    yield session_factory

    with engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE TABLE "
                + ", ".join(tables)
                + " RESTART IDENTITY CASCADE"
            )
        )

    database.clear_database_resource_cache()




@pytest.fixture
def integration_app(
    tmp_path,
    monkeypatch,
):
    """
    Create RobBotGPT FastAPI application using
    an isolated temporary SQLite database.

    External agent and RAG infrastructure are replaced
    because these tests focus on the HTTP + database layer.
    """

    # =====================================================
    # Temporary working directory
    # =====================================================

    monkeypatch.chdir(tmp_path)

    # =====================================================
    # Temporary SQLite database
    # =====================================================

    test_db_path = (
        tmp_path
        / "integration_test.db"
    )

    test_database_url = (
        f"sqlite:///{test_db_path}"
    )

    database.clear_database_resource_cache()

    monkeypatch.setattr(
        database.settings,
        "database_url",
        test_database_url,
    )

    test_engine = (
        database.get_engine()
    )

    database.Base.metadata.create_all(
        bind=test_engine
    )

    # =====================================================
    # Fake agent module
    # =====================================================

    fake_agent = ModuleType(
        "agent"
    )

    fake_agent.get_agent = (
        lambda *args, **kwargs: None
    )

    fake_agent.delete_thread_checkpoints = (
        lambda *args, **kwargs: None
    )

    fake_agent.clear_agent_cache = (
        lambda: None
    )

    monkeypatch.setitem(
        sys.modules,
        "agent",
        fake_agent,
    )

    # =====================================================
    # Fake RAG module
    # =====================================================

    fake_rag = ModuleType(
        "rag"
    )

    fake_rag.add_document_to_rag = (
        lambda *args, **kwargs: {
            "chunks": 0
        }
    )

    fake_rag.delete_document_from_rag = (
        lambda *args, **kwargs: {
            "deleted_chunks": 0,
            "deleted_files": 0,
        }
    )

    fake_rag.delete_thread_documents = (
        lambda *args, **kwargs: {
            "deleted_chunks": 0,
            "deleted_files": 0,
        }
    )

    fake_rag.clear_rag_resource_cache = (
        lambda: None
    )

    monkeypatch.setitem(
        sys.modules,
        "rag",
        fake_rag,
    )

    # =====================================================
    # Fresh application import
    # =====================================================

    sys.modules.pop(
        "app",
        None,
    )

    app_module = importlib.import_module(
        "app"
    )

    yield app_module

    # =====================================================
    # Cleanup
    # =====================================================

    test_engine = (
        database.get_engine()
    )

    database.Base.metadata.drop_all(
        bind=test_engine
    )

    database.clear_database_resource_cache()

    sys.modules.pop(
        "app",
        None,
    )


@pytest.fixture
def api_client(
    integration_app,
):
    with TestClient(
        integration_app.app
    ) as client:

        yield client





@pytest.fixture
def postgres_integration_app(
    tmp_path,
    monkeypatch,
    postgres_database,
    postgres_schema,
):
    """
    Create FastAPI application backed by the real
    PostgreSQL test database.

    External agent/RAG dependencies remain mocked,
    because this fixture verifies HTTP + PostgreSQL
    persistence.
    """

    monkeypatch.chdir(
        tmp_path
    )

    database.clear_database_resource_cache()

    monkeypatch.setattr(
        database.settings,
        "database_url",
        postgres_schema,
    )

    # ============================================
    # Fake agent
    # ============================================

    fake_agent = ModuleType(
        "agent"
    )

    fake_agent.get_agent = (
        lambda *args, **kwargs: None
    )

    fake_agent.delete_thread_checkpoints = (
        lambda *args, **kwargs: None
    )

    fake_agent.clear_agent_cache = (
        lambda: None
    )

    monkeypatch.setitem(
        sys.modules,
        "agent",
        fake_agent,
    )

    # ============================================
    # Fake RAG
    # ============================================

    fake_rag = ModuleType(
        "rag"
    )

    fake_rag.add_document_to_rag = (
        lambda *args, **kwargs: {
            "chunks": 1
        }
    )

    fake_rag.delete_document_from_rag = (
        lambda *args, **kwargs: {
            "deleted_chunks": 0,
            "deleted_files": 0,
        }
    )

    fake_rag.delete_thread_documents = (
        lambda *args, **kwargs: {
            "deleted_chunks": 0,
            "deleted_files": 0,
        }
    )

    fake_rag.clear_rag_resource_cache = (
        lambda: None
    )

    monkeypatch.setitem(
        sys.modules,
        "rag",
        fake_rag,
    )

    # ============================================
    # Fresh app import
    # ============================================

    sys.modules.pop(
        "app",
        None,
    )

    app_module = importlib.import_module(
        "app"
    )

    yield app_module

    database.clear_database_resource_cache()

    sys.modules.pop(
        "app",
        None,
    )




@pytest.fixture
def postgres_api_client(
    postgres_integration_app,
):
    with TestClient(
        postgres_integration_app.app
    ) as client:

        yield client