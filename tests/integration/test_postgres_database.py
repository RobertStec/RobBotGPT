import json
import pytest
import database

from database import (
    ChatMessage,
    Conversation,
    LongTermMemory,
    create_user,
    conversation_exists,
    create_or_update_conversation,
    get_chat_history,
    list_conversations,
    save_chat_message,
    save_memory,
    search_memory,
    delete_conversation,
)

from sqlalchemy import text




def _create_test_user(
    email: str,
):
    return create_user(
        email=email,
        password_hash="test-hash",
    )



def test_postgres_database_uses_real_postgresql(
    postgres_database,
):
    engine = database.get_engine()

    assert (
        engine.url.get_backend_name()
        == "postgresql"
    )

    db = postgres_database()

    try:
        result = db.execute(
            text(
                "SELECT current_database()"
            )
        ).scalar_one()

        assert result.endswith(
            "_test"
        )

    finally:
        db.close()





def test_postgres_conversation_crud(
    postgres_database,
):
    user = _create_test_user(
        "postgres-conversation@example.com"
    )

    thread_id = "postgres-conversation"

    assert conversation_exists(
        thread_id,
        user_id=user.id,
    ) is False

    create_or_update_conversation(
        thread_id=thread_id,
        first_message="Hello PostgreSQL",
        user_id=user.id,
    )

    assert conversation_exists(
        thread_id,
        user_id=user.id,
    ) is True

    db = postgres_database()

    try:
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id
                == thread_id
            )
            .one()
        )

        assert (
            conversation.title
            == "Hello PostgreSQL"
        )

    finally:
        db.close()

    # Calling create/update again must not
    # create another conversation.
    create_or_update_conversation(
        thread_id=thread_id,
        first_message="Different title",
        user_id=user.id,
    )

    db = postgres_database()

    try:
        count = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id
                == thread_id
            )
            .count()
        )

        assert count == 1

    finally:
        db.close()

    conversations = list_conversations(
        user_id=user.id,
    )

    assert len(conversations) == 1
    assert (
        conversations[0].thread_id
        == thread_id
    )
    assert conversation.user_id == user.id





def test_postgres_chat_messages_and_sources(
    postgres_database,
):
    user = _create_test_user(
        "postgres-chat@example.com"
    )

    thread_id = "postgres-chat"

    create_or_update_conversation(
        thread_id,
        "Document question",
        user_id=user.id,
    )

    save_chat_message(
        thread_id=thread_id,
        role="user",
        content="What is in the document?",
        user_id=user.id,
    )

    sources = [
        {
            "source": "manual.pdf",
            "page": 3,
            "chunk_index": 7,
        }
    ]

    save_chat_message(
        thread_id=thread_id,
        role="assistant",
        content="Document answer",
        sources=sources,
        user_id=user.id,
    )

    history = get_chat_history(
        thread_id,
        user_id=user.id,
    )

    assert len(history) == 2

    assert [
        message.role
        for message in history
    ] == [
        "user",
        "assistant",
    ]

    assert [
        message.content
        for message in history
    ] == [
        "What is in the document?",
        "Document answer",
    ]

    stored_sources = json.loads(
        history[1].sources
    )

    assert stored_sources == sources

    db = postgres_database()

    try:
        count = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.thread_id
                == thread_id,
                ChatMessage.user_id
                == user.id,
            )
            .count()
        )

        assert count == 2

    finally:
        db.close()





def test_postgres_long_term_memory_crud(
    postgres_database,
):
    user = _create_test_user(
        "postgres-memory@example.com"
    )

    thread_id = "postgres-memory"

    first_result = save_memory(
        thread_id,
        "I prefer Python.",
        user_id=user.id,
    )

    second_result = save_memory(
        thread_id,
        "I use LangGraph.",
        user_id=user.id,
    )

    assert (
        first_result
        == "Memory saved successfully."
    )

    assert (
        second_result
        == "Memory saved successfully."
    )

    result = search_memory(
        thread_id,
        "programming",
        user_id=user.id,
    )

    assert "I prefer Python." in result
    assert "I use LangGraph." in result

    db = postgres_database()

    try:
        memories = (
            db.query(LongTermMemory)
            .filter(
                LongTermMemory.thread_id
                == thread_id,
                LongTermMemory.user_id
                == user.id,
            )
            .all()
        )

        assert len(memories) == 2

    finally:
        db.close()





def test_postgres_thread_data_is_isolated(
    postgres_database,
):
    user = _create_test_user(
        "postgres-threads@example.com"
    )

    create_or_update_conversation(
        "thread-A",
        "Conversation A",
        user_id=user.id,
    )

    create_or_update_conversation(
        "thread-B",
        "Conversation B",
        user_id=user.id,
    )

    save_chat_message(
        "thread-A",
        "user",
        "Message A",
        user_id=user.id,
    )

    save_chat_message(
        "thread-B",
        "user",
        "Message B",
        user_id=user.id,
    )

    save_memory(
        "thread-A",
        "Memory A",
        user_id=user.id,
    )

    save_memory(
        "thread-B",
        "Memory B",
        user_id=user.id,
    )

    history_a = get_chat_history(
        "thread-A",
        user_id=user.id,
    )

    history_b = get_chat_history(
        "thread-B",
        user_id=user.id,
    )

    assert [
        message.content
        for message in history_a
    ] == [
        "Message A"
    ]

    assert [
        message.content
        for message in history_b
    ] == [
        "Message B"
    ]

    memory_a = search_memory(
        "thread-A",
        "anything",
        user_id=user.id,
    )

    memory_b = search_memory(
        "thread-B",
        "anything",
        user_id=user.id,
    )

    assert "Memory A" in memory_a
    assert "Memory B" not in memory_a

    assert "Memory B" in memory_b
    assert "Memory A" not in memory_b





def test_postgres_delete_conversation_removes_related_data(
    postgres_database,
):
    user = _create_test_user(
        "postgres-delete@example.com"
    )

    thread_id = "postgres-delete"

    create_or_update_conversation(
        thread_id,
        "Delete me",
        user_id=user.id,
    )

    save_chat_message(
        thread_id,
        "user",
        "Message",
        user_id=user.id,
    )

    save_memory(
        thread_id,
        "Memory",
        user_id=user.id,
    )

    result = delete_conversation(
        thread_id,
        user_id=user.id,
    )

    assert result is True

    db = postgres_database()

    try:
        conversation_count = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id
                == thread_id,
                Conversation.user_id
                == user.id,
            )
            .count()
        )

        message_count = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.thread_id
                == thread_id,
                ChatMessage.user_id
                == user.id,
            )
            .count()
        )

        memory_count = (
            db.query(LongTermMemory)
            .filter(
                LongTermMemory.thread_id
                == thread_id,
                LongTermMemory.user_id
                == user.id,
            )
            .count()
        )

        assert conversation_count == 0
        assert message_count == 0
        assert memory_count == 0

    finally:
        db.close()





def test_postgres_delete_conversation_rolls_back_on_commit_failure(
    postgres_database,
    monkeypatch,
    mocker,
):
    user = _create_test_user(
        "postgres-rollback@example.com"
    )

    thread_id = "postgres-rollback"

    create_or_update_conversation(
        thread_id,
        "Rollback test",
        user_id=user.id,
    )

    save_chat_message(
        thread_id,
        "user",
        "Important message",
        user_id=user.id,
    )

    save_memory(
        thread_id,
        "Important memory",
        user_id=user.id,
    )

    failing_session = (
        postgres_database()
    )

    rollback_spy = mocker.spy(
        failing_session,
        "rollback",
    )

    mocker.patch.object(
        failing_session,
        "commit",
        side_effect=RuntimeError(
            "forced commit failure"
        ),
    )

    with monkeypatch.context() as context:
        context.setattr(
            database,
            "create_session",
            lambda: failing_session,
        )

        with pytest.raises(
            RuntimeError,
            match="forced commit failure",
        ):
            delete_conversation(
                thread_id,
                user_id=user.id,
            )

    rollback_spy.assert_called_once()

    # Open a new real PostgreSQL session.
    db = postgres_database()

    try:
        conversation_count = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id
                == thread_id,
                Conversation.user_id
                == user.id,
            )
            .count()
        )

        message_count = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.thread_id
                == thread_id,
                ChatMessage.user_id
                == user.id,
            )
            .count()
        )

        memory_count = (
            db.query(LongTermMemory)
            .filter(
                LongTermMemory.thread_id
                == thread_id,
                LongTermMemory.user_id
                == user.id,
            )
            .count()
        )

        assert conversation_count == 1
        assert message_count == 1
        assert memory_count == 1

    finally:
        db.close()