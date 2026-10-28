from datetime import (
    datetime,
    timedelta,
)

import database

import pytest

from sqlalchemy.exc import IntegrityError

from database import (
    AuthSession,
    ChatMessage,
    Conversation,
    LongTermMemory,
    User,
    RagDocument,
    conversation_exists,
    create_auth_session,
    create_or_update_conversation,
    create_user,
    delete_auth_session,
    delete_conversation,
    get_auth_session_by_token_hash,
    get_chat_history,
    get_user_by_email,
    get_user_by_id,
    list_conversations,
    save_chat_message,
    save_memory,
    search_memory,
)



# =========================================================
# Conversation ownership
# =========================================================


def _create_test_user(
    email: str,
):
    return create_user(
        email=email,
        password_hash="test-hash",
    )


def test_create_conversation(
    test_database,
):
    user = _create_test_user(
        "owner@example.com"
    )

    result = create_or_update_conversation(
        thread_id="thread-001",
        first_message="Hello RobBotGPT",
        user_id=user.id,
    )

    assert result is True

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
        assert conversation.user_id == user.id
        assert conversation.thread_id == "thread-001"
        assert conversation.title == "Hello RobBotGPT"

    finally:
        db.close()


def test_create_conversation_truncates_long_title(
    test_database,
):
    user = _create_test_user(
        "title@example.com"
    )

    long_message = "A" * 60

    create_or_update_conversation(
        thread_id="thread-long-title",
        first_message=long_message,
        user_id=user.id,
    )

    conversations = list_conversations(
        user_id=user.id
    )

    assert len(conversations) == 1

    assert conversations[0].title == (
        "A" * 40 + "..."
    )


def test_create_or_update_conversation_does_not_duplicate(
    test_database,
):
    user = _create_test_user(
        "duplicate@example.com"
    )

    first_result = create_or_update_conversation(
        thread_id="thread-001",
        first_message="First message",
        user_id=user.id,
    )

    second_result = create_or_update_conversation(
        thread_id="thread-001",
        first_message="Different message",
        user_id=user.id,
    )

    assert first_result is True
    assert second_result is True

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


def test_conversation_cannot_be_claimed_by_another_user(
    test_database,
):
    owner = _create_test_user(
        "owner@example.com"
    )

    other_user = _create_test_user(
        "other@example.com"
    )

    first_result = create_or_update_conversation(
        thread_id="shared-thread",
        first_message="Owner conversation",
        user_id=owner.id,
    )

    second_result = create_or_update_conversation(
        thread_id="shared-thread",
        first_message="Attempted takeover",
        user_id=other_user.id,
    )

    assert first_result is True
    assert second_result is False

    db = test_database()

    try:
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id
                == "shared-thread"
            )
            .first()
        )

        assert conversation is not None
        assert conversation.user_id == owner.id
        assert conversation.title == "Owner conversation"

    finally:
        db.close()


@pytest.mark.parametrize(
    (
        "model_class",
        "kwargs",
    ),
    [
        (
            Conversation,
            {
                "thread_id": "legacy-conversation",
                "title": "Legacy",
            },
        ),
        (
            ChatMessage,
            {
                "thread_id": "legacy-message",
                "role": "user",
                "content": "Legacy message",
            },
        ),
        (
            LongTermMemory,
            {
                "thread_id": "legacy-memory",
                "memory": "Legacy memory",
            },
        ),
        (
            RagDocument,
            {
                "thread_id": "legacy-rag",
                "source_name": "legacy.pdf",
                "stored_name": "legacy.txt",
            },
        ),
    ],
)


def test_user_owned_records_require_user_id(
    test_database,
    model_class,
    kwargs,
):
    db = test_database()

    try:
        record = model_class(
            user_id=None,
            **kwargs,
        )

        db.add(record)

        with pytest.raises(
            IntegrityError
        ):
            db.commit()

        db.rollback()

    finally:
        db.close()



def test_list_conversations_is_isolated_by_user(
    test_database,
):
    user_a = _create_test_user(
        "user-a@example.com"
    )

    user_b = _create_test_user(
        "user-b@example.com"
    )

    create_or_update_conversation(
        thread_id="thread-A",
        first_message="Conversation A",
        user_id=user_a.id,
    )

    create_or_update_conversation(
        thread_id="thread-B",
        first_message="Conversation B",
        user_id=user_b.id,
    )

    conversations_a = list_conversations(
        user_id=user_a.id
    )

    conversations_b = list_conversations(
        user_id=user_b.id
    )

    assert [
        item.thread_id
        for item in conversations_a
    ] == [
        "thread-A"
    ]

    assert [
        item.thread_id
        for item in conversations_b
    ] == [
        "thread-B"
    ]


# =========================================================
# Chat messages
# =========================================================


def test_save_chat_message(
    test_database,
):
    user = _create_test_user(
        "chat@example.com"
    )

    create_or_update_conversation(
        thread_id="thread-001",
        first_message="Hello",
        user_id=user.id,
    )

    result = save_chat_message(
        thread_id="thread-001",
        role="user",
        content="Hello RobBotGPT",
        user_id=user.id,
    )

    assert result is True

    history = get_chat_history(
        "thread-001",
        user_id=user.id,
    )

    assert len(history) == 1
    assert history[0].user_id == user.id
    assert history[0].role == "user"
    assert history[0].content == "Hello RobBotGPT"


def test_save_chat_message_rejects_foreign_conversation(
    test_database,
):
    owner = _create_test_user(
        "message-owner@example.com"
    )

    other_user = _create_test_user(
        "message-other@example.com"
    )

    create_or_update_conversation(
        thread_id="owner-thread",
        first_message="Owner",
        user_id=owner.id,
    )

    result = save_chat_message(
        thread_id="owner-thread",
        role="user",
        content="Unauthorized message",
        user_id=other_user.id,
    )

    assert result is False

    owner_history = get_chat_history(
        "owner-thread",
        user_id=owner.id,
    )

    assert owner_history == []


def test_get_chat_history_returns_messages_in_order(
    test_database,
):
    user = _create_test_user(
        "history@example.com"
    )

    create_or_update_conversation(
        thread_id="thread-001",
        first_message="Question",
        user_id=user.id,
    )

    save_chat_message(
        "thread-001",
        "user",
        "First message",
        user_id=user.id,
    )

    save_chat_message(
        "thread-001",
        "assistant",
        "Second message",
        user_id=user.id,
    )

    history = get_chat_history(
        "thread-001",
        user_id=user.id,
    )

    assert [
        message.content
        for message in history
    ] == [
        "First message",
        "Second message",
    ]


def test_save_chat_message_serializes_sources(
    test_database,
):
    user = _create_test_user(
        "rag-message@example.com"
    )

    create_or_update_conversation(
        thread_id="thread-rag",
        first_message="Document question",
        user_id=user.id,
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
        user_id=user.id,
    )

    history = get_chat_history(
        "thread-rag",
        user_id=user.id,
    )

    assert len(history) == 1
    assert history[0].sources is not None
    assert '"document.pdf"' in history[0].sources
    assert '"page": 3' in history[0].sources


def test_chat_history_is_isolated_by_user(
    test_database,
):
    owner = _create_test_user(
        "history-owner@example.com"
    )

    other_user = _create_test_user(
        "history-other@example.com"
    )

    create_or_update_conversation(
        thread_id="private-thread",
        first_message="Private conversation",
        user_id=owner.id,
    )

    save_chat_message(
        thread_id="private-thread",
        role="user",
        content="Private message",
        user_id=owner.id,
    )

    owner_history = get_chat_history(
        "private-thread",
        user_id=owner.id,
    )

    other_history = get_chat_history(
        "private-thread",
        user_id=other_user.id,
    )

    assert len(owner_history) == 1
    assert owner_history[0].content == "Private message"

    assert other_history == []


# =========================================================
# Long-term memory
# =========================================================


def test_save_memory(
    test_database,
):
    user = _create_test_user(
        "memory@example.com"
    )

    result = save_memory(
        thread_id="thread-memory",
        memory="I prefer Python.",
        user_id=user.id,
    )

    assert result == "Memory saved successfully."

    db = test_database()

    try:
        memory = (
            db.query(LongTermMemory)
            .filter(
                LongTermMemory.thread_id
                == "thread-memory",
                LongTermMemory.user_id
                == user.id,
            )
            .first()
        )

        assert memory is not None
        assert memory.user_id == user.id
        assert memory.memory == "I prefer Python."

    finally:
        db.close()



def test_search_memory_returns_saved_memories(
    test_database,
):
    user = _create_test_user(
        "memory-search@example.com"
    )

    save_memory(
        "thread-memory",
        "I prefer Python.",
        user_id=user.id,
    )

    save_memory(
        "thread-memory",
        "I use LangGraph.",
        user_id=user.id,
    )

    result = search_memory(
        thread_id="thread-memory",
        query="programming",
        user_id=user.id,
    )

    assert "I prefer Python." in result
    assert "I use LangGraph." in result




def test_search_memory_is_isolated_by_user(
    test_database,
):
    user_a = _create_test_user(
        "memory-a@example.com"
    )

    user_b = _create_test_user(
        "memory-b@example.com"
    )

    save_memory(
        thread_id="shared-memory-thread",
        memory="Memory from A",
        user_id=user_a.id,
    )

    save_memory(
        thread_id="shared-memory-thread",
        memory="Memory from B",
        user_id=user_b.id,
    )

    result_a = search_memory(
        thread_id="shared-memory-thread",
        query="anything",
        user_id=user_a.id,
    )

    result_b = search_memory(
        thread_id="shared-memory-thread",
        query="anything",
        user_id=user_b.id,
    )

    assert "Memory from A" in result_a
    assert "Memory from B" not in result_a

    assert "Memory from B" in result_b
    assert "Memory from A" not in result_b


def test_search_memory_returns_message_when_empty(
    test_database,
):
    user = _create_test_user(
        "empty-memory@example.com"
    )

    result = search_memory(
        "unknown-thread",
        "anything",
        user_id=user.id,
    )

    assert result == "No saved memory found."


# =========================================================
# Conversation deletion
# =========================================================


def test_delete_conversation_removes_related_data(
    test_database,
):
    user = _create_test_user(
        "delete-owner@example.com"
    )

    create_or_update_conversation(
        thread_id="thread-delete",
        first_message="Delete me",
        user_id=user.id,
    )

    save_chat_message(
        "thread-delete",
        "user",
        "Message",
        user_id=user.id,
    )

    save_memory(
        "thread-delete",
        "Memory",
        user_id=user.id,
    )

    result = delete_conversation(
        "thread-delete",
        user_id=user.id,
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


def test_delete_conversation_rejects_foreign_user(
    test_database,
):
    owner = _create_test_user(
        "delete-owner@example.com"
    )

    other_user = _create_test_user(
        "delete-other@example.com"
    )

    create_or_update_conversation(
        thread_id="private-delete-thread",
        first_message="Private",
        user_id=owner.id,
    )

    save_chat_message(
        thread_id="private-delete-thread",
        role="user",
        content="Private message",
        user_id=owner.id,
    )

    result = delete_conversation(
        "private-delete-thread",
        user_id=other_user.id,
    )

    assert result is False

    assert conversation_exists(
        "private-delete-thread",
        user_id=owner.id,
    ) is True

    history = get_chat_history(
        "private-delete-thread",
        user_id=owner.id,
    )

    assert len(history) == 1
    assert history[0].content == "Private message"


def test_delete_conversation_returns_false_when_not_found(
    test_database,
):
    user = _create_test_user(
        "delete-missing@example.com"
    )

    result = delete_conversation(
        "does-not-exist",
        user_id=user.id,
    )

    assert result is False



# =========================================================
# Authentication
# =========================================================


def test_create_user(
    test_database,
):
    user = create_user(
        email="user@example.com",
        password_hash="hashed-password",
    )

    assert user.id is not None
    assert user.email == "user@example.com"
    assert user.password_hash == "hashed-password"
    assert user.is_active is True

    db = test_database()

    try:
        stored_user = (
            db.query(User)
            .filter(
                User.email
                == "user@example.com"
            )
            .first()
        )

        assert stored_user is not None

    finally:
        db.close()




def test_get_user_by_email(
    test_database,
):
    created = create_user(
        email="email@example.com",
        password_hash="hash",
    )

    user = get_user_by_email(
        "email@example.com"
    )

    assert user is not None
    assert user.id == created.id
    assert user.email == "email@example.com"




def test_get_user_by_email_returns_none_when_missing(
    test_database,
):
    user = get_user_by_email(
        "missing@example.com"
    )

    assert user is None




def test_get_user_by_id(
    test_database,
):
    created = create_user(
        email="id@example.com",
        password_hash="hash",
    )

    user = get_user_by_id(
        created.id
    )

    assert user is not None
    assert user.id == created.id
    assert user.email == "id@example.com"




def test_create_auth_session(
    test_database,
):
    user = create_user(
        email="session@example.com",
        password_hash="hash",
    )

    expires_at = (
        datetime.utcnow()
        + timedelta(hours=24)
    )

    auth_session = create_auth_session(
        user_id=user.id,
        token_hash="a" * 64,
        expires_at=expires_at,
    )

    assert auth_session.id is not None
    assert auth_session.user_id == user.id
    assert auth_session.token_hash == "a" * 64

    db = test_database()

    try:
        stored_session = (
            db.query(AuthSession)
            .filter(
                AuthSession.id
                == auth_session.id
            )
            .first()
        )

        assert stored_session is not None

    finally:
        db.close()




def test_get_auth_session_by_token_hash(
    test_database,
):
    user = create_user(
        email="lookup@example.com",
        password_hash="hash",
    )

    token_hash = "b" * 64

    created = create_auth_session(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=(
            datetime.utcnow()
            + timedelta(hours=24)
        ),
    )

    auth_session = (
        get_auth_session_by_token_hash(
            token_hash
        )
    )

    assert auth_session is not None
    assert auth_session.id == created.id
    assert auth_session.user_id == user.id




def test_get_auth_session_returns_none_when_missing(
    test_database,
):
    auth_session = (
        get_auth_session_by_token_hash(
            "c" * 64
        )
    )

    assert auth_session is None




def test_delete_auth_session(
    test_database,
):
    user = create_user(
        email="delete@example.com",
        password_hash="hash",
    )

    token_hash = "d" * 64

    create_auth_session(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=(
            datetime.utcnow()
            + timedelta(hours=24)
        ),
    )

    result = delete_auth_session(
        token_hash
    )

    assert result is True

    assert (
        get_auth_session_by_token_hash(
            token_hash
        )
        is None
    )




def test_delete_auth_session_returns_false_when_missing(
    test_database,
):
    result = delete_auth_session(
        "e" * 64
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
    user = _create_test_user(
        "exists@example.com"
    )

    other_user = _create_test_user(
        "exists-other@example.com"
    )

    assert (
        database.conversation_exists(
            "thread-existing",
            user_id=user.id,
        )
        is False
    )

    create_or_update_conversation(
        thread_id="thread-existing",
        first_message="Hello",
        user_id=user.id,
    )

    assert (
        database.conversation_exists(
            "thread-existing",
            user_id=user.id,
        )
        is True
    )

    assert (
        database.conversation_exists(
            "thread-existing",
            user_id=other_user.id,
        )
        is False
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