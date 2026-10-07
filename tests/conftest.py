import pytest
import database

import importlib
import sys
from types import ModuleType

from fastapi.testclient import TestClient


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