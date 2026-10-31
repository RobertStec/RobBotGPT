import database


PASSWORD = "StrongPassword123!"


def _register_and_login(
    api_client,
    email: str,
):
    register_response = api_client.post(
        "/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
        },
    )

    assert register_response.status_code == 201

    login_response = api_client.post(
        "/auth/login",
        json={
            "email": email,
            "password": PASSWORD,
        },
    )

    assert login_response.status_code == 200

    return login_response.json()["user"]


def test_history_requires_authentication(
    api_client,
):
    response = api_client.get(
        "/history/thread-001"
    )

    assert response.status_code == 401

    assert response.json()["error_code"] == (
        "authentication_required"
    )


def test_history_returns_404_for_unknown_thread(
    api_client,
):
    _register_and_login(
        api_client,
        "unknown@example.com",
    )

    response = api_client.get(
        "/history/unknown-thread"
    )

    assert response.status_code == 404

    assert response.json() == {
        "success": False,
        "message": "Conversation not found.",
        "error_code": "conversation_not_found",
    }


def test_history_returns_saved_messages(
    api_client,
):
    user = _register_and_login(
        api_client,
        "history@example.com",
    )

    database.create_or_update_conversation(
        thread_id="thread-history",
        first_message="Hello",
        user_id=user["id"],
    )

    database.save_chat_message(
        thread_id="thread-history",
        role="user",
        content="Hello RobBotGPT",
        user_id=user["id"],
    )

    database.save_chat_message(
        thread_id="thread-history",
        role="assistant",
        content="Hello!",
        user_id=user["id"],
    )

    response = api_client.get(
        "/history/thread-history"
    )

    assert response.status_code == 200

    assert response.json()["messages"] == [
        {
            "role": "user",
            "content": "Hello RobBotGPT",
            "sources": [],
        },
        {
            "role": "assistant",
            "content": "Hello!",
            "sources": [],
        },
    ]


def test_history_deserializes_rag_sources(
    api_client,
):
    user = _register_and_login(
        api_client,
        "rag-history@example.com",
    )

    database.create_or_update_conversation(
        thread_id="thread-rag",
        first_message="Question about document",
        user_id=user["id"],
    )

    sources = [
        {
            "source": "manual.pdf",
            "page": 4,
            "chunk_index": 8,
        }
    ]

    database.save_chat_message(
        thread_id="thread-rag",
        role="assistant",
        content="Answer from document.",
        sources=sources,
        user_id=user["id"],
    )

    response = api_client.get(
        "/history/thread-rag"
    )

    assert response.status_code == 200

    message = (
        response.json()["messages"][0]
    )

    assert (
        message["content"]
        == "Answer from document."
    )

    assert message["sources"] == [
        {
            "source": "manual.pdf",
            "page": 4,
            "chunk_index": 8,
        }
    ]


def test_history_of_another_user_returns_404(
    api_client,
):
    owner = _register_and_login(
        api_client,
        "history-owner@example.com",
    )

    database.create_or_update_conversation(
        thread_id="private-history",
        first_message="Private conversation",
        user_id=owner["id"],
    )

    database.save_chat_message(
        thread_id="private-history",
        role="user",
        content="Private message",
        user_id=owner["id"],
    )

    api_client.post(
        "/auth/logout"
    )

    _register_and_login(
        api_client,
        "history-other@example.com",
    )

    response = api_client.get(
        "/history/private-history"
    )

    assert response.status_code == 404

    assert response.json()["error_code"] == (
        "conversation_not_found"
    )


def test_history_rejects_invalid_thread_id(
    api_client,
):
    _register_and_login(
        api_client,
        "history-invalid-thread@example.com",
    )

    response = api_client.get(
        "/history/bad%20thread"
    )

    assert response.status_code == 422

    assert response.json() == {
        "success": False,
        "message": "Request validation failed.",
        "error_code": "request_validation_failed",
    }