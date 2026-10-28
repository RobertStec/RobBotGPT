import importlib

from langchain_core.messages import (
    AIMessageChunk,
)

import services.document_service as document_service


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


def _logout(api_client):
    response = api_client.post(
        "/auth/logout"
    )

    assert response.status_code == 200


def _conversation_ids(response):
    assert response.status_code == 200

    return {
        conversation["thread_id"]
        for conversation
        in response.json()["conversations"]
    }


def test_two_users_are_fully_isolated(
    api_client,
    integration_app,
    mocker,
):
    chat_router = importlib.import_module(
        "api.routers.chat"
    )

    fake_agent = mocker.Mock()

    fake_agent.stream.return_value = [
        {
            "type": "messages",
            "data": (
                AIMessageChunk(
                    content="Assistant response."
                ),
                {
                    "langgraph_node": "chatbot"
                },
            ),
        }
    ]

    mock_get_agent = mocker.patch.object(
        chat_router,
        "get_agent",
        return_value=fake_agent,
    )

    # =====================================================
    # USER A
    # =====================================================

    user_a = _register_and_login(
        api_client,
        "isolation-a@example.com",
    )

    response_a = api_client.post(
        "/chat/stream",
        json={
            "message": "Private message A",
            "thread_id": "private-thread-a",
            "model": "gpt-4o-mini",
        },
    )

    assert response_a.status_code == 200

    conversations_a = api_client.get(
        "/conversations"
    )

    assert _conversation_ids(
        conversations_a
    ) == {
        "private-thread-a"
    }

    history_a = api_client.get(
        "/history/private-thread-a"
    )

    assert history_a.status_code == 200

    messages_a = (
        history_a.json()["messages"]
    )

    assert len(messages_a) == 2

    assert (
        messages_a[0]["content"]
        == "Private message A"
    )

    # =====================================================
    # USER B
    # =====================================================

    _logout(api_client)

    user_b = _register_and_login(
        api_client,
        "isolation-b@example.com",
    )

    assert user_a["id"] != user_b["id"]

    response_b = api_client.post(
        "/chat/stream",
        json={
            "message": "Private message B",
            "thread_id": "private-thread-b",
            "model": "gpt-4o-mini",
        },
    )

    assert response_b.status_code == 200

    conversations_b = api_client.get(
        "/conversations"
    )

    assert _conversation_ids(
        conversations_b
    ) == {
        "private-thread-b"
    }

    history_b = api_client.get(
        "/history/private-thread-b"
    )

    assert history_b.status_code == 200

    assert (
        history_b.json()["messages"][0][
            "content"
        ]
        == "Private message B"
    )

    # =====================================================
    # USER B CANNOT READ USER A
    # =====================================================

    foreign_history = api_client.get(
        "/history/private-thread-a"
    )

    assert foreign_history.status_code == 404

    assert (
        foreign_history.json()["error_code"]
        == "conversation_not_found"
    )

    # =====================================================
    # USER B CANNOT WRITE TO USER A THREAD
    # =====================================================

    get_agent_calls_before = (
        mock_get_agent.call_count
    )

    foreign_chat = api_client.post(
        "/chat/stream",
        json={
            "message": (
                "Try to write into user A thread"
            ),
            "thread_id": "private-thread-a",
            "model": "gpt-4o-mini",
        },
    )

    assert foreign_chat.status_code == 404

    assert (
        foreign_chat.json()["error_code"]
        == "conversation_not_found"
    )

    assert (
        mock_get_agent.call_count
        == get_agent_calls_before
    )

    # =====================================================
    # USER B CANNOT UPLOAD TO USER A THREAD
    # =====================================================

    mock_add_document = mocker.patch.object(
        document_service,
        "add_document_to_rag",
    )

    foreign_upload = api_client.post(
        "/upload",
        data={
            "thread_id": "private-thread-a"
        },
        files={
            "file": (
                "secret.txt",
                b"Unauthorized content",
                "text/plain",
            )
        },
    )

    assert foreign_upload.status_code == 404

    assert (
        foreign_upload.json()["error_code"]
        == "conversation_not_found"
    )

    mock_add_document.assert_not_called()

    # =====================================================
    # USER B CANNOT DELETE USER A
    # =====================================================

    foreign_delete = api_client.delete(
        "/conversations/private-thread-a"
    )

    assert foreign_delete.status_code == 404

    assert (
        foreign_delete.json()["error_code"]
        == "conversation_not_found"
    )

    # =====================================================
    # USER A DATA STILL EXISTS
    # =====================================================

    _logout(api_client)

    login_a = api_client.post(
        "/auth/login",
        json={
            "email": "isolation-a@example.com",
            "password": PASSWORD,
        },
    )

    assert login_a.status_code == 200

    conversations_a_after = api_client.get(
        "/conversations"
    )

    assert _conversation_ids(
        conversations_a_after
    ) == {
        "private-thread-a"
    }

    history_a_after = api_client.get(
        "/history/private-thread-a"
    )

    assert history_a_after.status_code == 200

    messages_a_after = (
        history_a_after.json()["messages"]
    )

    assert len(messages_a_after) == 2

    assert (
        messages_a_after[0]["content"]
        == "Private message A"
    )

    assert (
        "private-thread-b"
        not in _conversation_ids(
            conversations_a_after
        )
    )