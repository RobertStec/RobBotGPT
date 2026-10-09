import database

from database import (
    ChatMessage,
    Conversation,
    LongTermMemory,
    create_or_update_conversation,
    delete_conversation,
    get_chat_history,
    list_conversations,
    save_chat_message,
    save_memory,
    search_memory,
)


# Tworzenie konwersacji

def test_create_conversation(
    test_database,
):
    create_or_update_conversation(
        thread_id="thread-001",
        first_message="Hello RobBotGPT",
    )

    db = test_database()

    try:
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id
                == "thread-001"
            )
            .first()
        )

        assert conversation is not None
        assert conversation.thread_id == "thread-001"
        assert conversation.title == "Hello RobBotGPT"

    finally:
        db.close()



# Test skracania tytułu

def test_create_conversation_truncates_long_title(
    test_database,
):
    long_message = "A" * 60

    create_or_update_conversation(
        thread_id="thread-long-title",
        first_message=long_message,
    )

    conversations = list_conversations()

    assert len(conversations) == 1
    assert conversations[0].title == (
        "A" * 40 + "..."
    )


# Ponowne wywołanie nie tworzy duplikatu

def test_create_or_update_conversation_does_not_duplicate(
    test_database,
):
    create_or_update_conversation(
        "thread-001",
        "First message",
    )

    create_or_update_conversation(
        "thread-001",
        "Different message",
    )

    db = test_database()

    try:
        count = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id
                == "thread-001"
            )
            .count()
        )

        assert count == 1

    finally:
        db.close()




# Zapis wiadomości

def test_save_chat_message(
    test_database,
):
    create_or_update_conversation(
        "thread-001",
        "Hello",
    )

    save_chat_message(
        thread_id="thread-001",
        role="user",
        content="Hello RobBotGPT",
    )

    history = get_chat_history(
        "thread-001"
    )

    assert len(history) == 1

    assert history[0].role == "user"
    assert history[0].content == "Hello RobBotGPT"



# Historia zachowuje kolejność

def test_get_chat_history_returns_messages_in_order(
    test_database,
):
    create_or_update_conversation(
        "thread-001",
        "Question",
    )

    save_chat_message(
        "thread-001",
        "user",
        "First message",
    )

    save_chat_message(
        "thread-001",
        "assistant",
        "Second message",
    )

    history = get_chat_history(
        "thread-001"
    )

    assert [
        message.content
        for message in history
    ] == [
        "First message",
        "Second message",
    ]



# Źródła RAG zapisują się jako JSON

def test_save_chat_message_serializes_sources(
    test_database,
):
    create_or_update_conversation(
        "thread-rag",
        "Document question",
    )

    sources = [
        {
            "source": "document.pdf",
            "page": 3,
            "chunk_index": 7,
        }
    ]

    save_chat_message(
        thread_id="thread-rag",
        role="assistant",
        content="Document answer",
        sources=sources,
    )

    history = get_chat_history(
        "thread-rag"
    )

    assert len(history) == 1

    assert history[0].sources is not None
    assert '"document.pdf"' in history[0].sources
    assert '"page": 3' in history[0].sources




# Izolacja historii przez thread_id

def test_chat_history_is_isolated_by_thread_id(
    test_database,
):
    create_or_update_conversation(
        "thread-A",
        "Conversation A",
    )

    create_or_update_conversation(
        "thread-B",
        "Conversation B",
    )

    save_chat_message(
        "thread-A",
        "user",
        "Message A",
    )

    save_chat_message(
        "thread-B",
        "user",
        "Message B",
    )

    history_a = get_chat_history(
        "thread-A"
    )

    history_b = get_chat_history(
        "thread-B"
    )

    assert len(history_a) == 1
    assert history_a[0].content == "Message A"

    assert len(history_b) == 1
    assert history_b[0].content == "Message B"



# Test save_memory

def test_save_memory(
    test_database,
):
    result = save_memory(
        thread_id="thread-memory",
        memory="I prefer Python.",
    )

    assert result == "Memory saved successfully."

    db = test_database()

    try:
        memory = (
            db.query(LongTermMemory)
            .filter(
                LongTermMemory.thread_id
                == "thread-memory"
            )
            .first()
        )

        assert memory is not None
        assert memory.memory == "I prefer Python."

    finally:
        db.close()



# Test search_memory

def test_search_memory_returns_saved_memories(
    test_database,
):
    save_memory(
        "thread-memory",
        "I prefer Python.",
    )

    save_memory(
        "thread-memory",
        "I use LangGraph.",
    )

    result = search_memory(
        thread_id="thread-memory",
        query="programming",
    )

    assert "I prefer Python." in result
    assert "I use LangGraph." in result



# Izolacja pamięci

def test_search_memory_is_isolated_by_thread_id(
    test_database,
):
    save_memory(
        "thread-A",
        "Memory from A",
    )

    save_memory(
        "thread-B",
        "Memory from B",
    )

    result_a = search_memory(
        "thread-A",
        "anything",
    )

    assert "Memory from A" in result_a
    assert "Memory from B" not in result_a



# Brak zapisanej pamięci

def test_search_memory_returns_message_when_empty(
    test_database,
):
    result = search_memory(
        "unknown-thread",
        "anything",
    )

    assert result == "No saved memory found."



# Usuwanie całej konwersacji

def test_delete_conversation_removes_related_data(
    test_database,
):
    create_or_update_conversation(
        "thread-delete",
        "Delete me",
    )

    save_chat_message(
        "thread-delete",
        "user",
        "Message",
    )

    save_memory(
        "thread-delete",
        "Memory",
    )

    result = delete_conversation(
        "thread-delete"
    )

    assert result is True

    db = test_database()

    try:
        conversation_count = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id
                == "thread-delete"
            )
            .count()
        )

        message_count = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.thread_id
                == "thread-delete"
            )
            .count()
        )

        memory_count = (
            db.query(LongTermMemory)
            .filter(
                LongTermMemory.thread_id
                == "thread-delete"
            )
            .count()
        )

        assert conversation_count == 0
        assert message_count == 0
        assert memory_count == 0

    finally:
        db.close()



# Usunięcie nieistniejącej konwersacji

def test_delete_conversation_returns_false_when_not_found(
    test_database,
):
    result = delete_conversation(
        "does-not-exist"
    )

    assert result is False



# =========================================================
# Database infrastructure
# =========================================================


def test_create_database_engine_uses_sqlite_connect_args(
    mocker,
    tmp_path,
):
    database_url = (
        f"sqlite:///{tmp_path / 'factory.db'}"
    )

    mock_create_engine = mocker.patch(
        "database.create_engine"
    )

    mock_engine = mocker.Mock()

    mock_create_engine.return_value = (
        mock_engine
    )

    result = (
        database.create_database_engine(
            database_url
        )
    )

    assert result is mock_engine

    mock_create_engine.assert_called_once_with(
        database_url,
        pool_pre_ping=True,
        connect_args={
            "check_same_thread": False
        },
    )




# Test PostgreSQL nie może dostać parametrów SQLite

def test_create_database_engine_does_not_use_sqlite_args_for_postgresql(
    mocker,
):
    database_url = (
        "postgresql://"
        "user:password@localhost:5432/"
        "robbotgpt"
    )

    mock_create_engine = mocker.patch(
        "database.create_engine"
    )

    mock_engine = mocker.Mock()

    mock_create_engine.return_value = (
        mock_engine
    )

    result = (
        database.create_database_engine(
            database_url
        )
    )

    assert result is mock_engine

    mock_create_engine.assert_called_once_with(
        database_url,
        pool_pre_ping=True,
    )



# Test lazy cache engine

def test_get_engine_caches_engine(
    mocker,
):
    database.clear_database_resource_cache()

    mock_engine = mocker.Mock()

    mock_create = mocker.patch(
        "database.create_database_engine",
        return_value=mock_engine,
    )

    first = database.get_engine(
        "sqlite:///cache_test.db"
    )

    second = database.get_engine(
        "sqlite:///cache_test.db"
    )

    assert first is mock_engine
    assert second is mock_engine

    mock_create.assert_called_once_with(
        "sqlite:///cache_test.db"
    )

    database.clear_database_resource_cache()




# Test różnych DB URL

def test_get_engine_keeps_separate_engines_per_database_url(
    mocker,
):
    database.clear_database_resource_cache()

    first_engine = mocker.Mock()
    second_engine = mocker.Mock()

    mocker.patch(
        "database.create_database_engine",
        side_effect=[
            first_engine,
            second_engine,
        ],
    )

    result_a = database.get_engine(
        "sqlite:///database_a.db"
    )

    result_b = database.get_engine(
        "sqlite:///database_b.db"
    )

    assert result_a is first_engine
    assert result_b is second_engine

    database.clear_database_resource_cache()




# Test cleanup

def test_clear_database_resource_cache_disposes_engines(
    mocker,
):
    database.clear_database_resource_cache()

    mock_engine = mocker.Mock()

    mocker.patch(
        "database.create_database_engine",
        return_value=mock_engine,
    )

    database.get_engine(
        "sqlite:///cleanup.db"
    )

    database.get_session_factory(
        "sqlite:///cleanup.db"
    )

    database.clear_database_resource_cache()

    mock_engine.dispose.assert_called_once()





# Test conversation_exists

def test_conversation_exists(
    test_database,
):
    assert (
        database.conversation_exists(
            "thread-existing"
        )
        is False
    )

    create_or_update_conversation(
        thread_id="thread-existing",
        first_message="Hello",
    )

    assert (
        database.conversation_exists(
            "thread-existing"
        )
        is True
    )



# Test database connection

def test_init_db_checks_database_connection(
    mocker,
):
    mock_engine = mocker.MagicMock()

    mock_connection = (
        mock_engine.connect.return_value
        .__enter__.return_value
    )

    mocker.patch(
        "database.get_engine",
        return_value=mock_engine,
    )

    database.init_db()

    mock_engine.connect.assert_called_once_with()

    mock_connection.execute.assert_called_once()