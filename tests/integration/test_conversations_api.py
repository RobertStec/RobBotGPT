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


def test_get_conversations_requires_authentication(
    api_client,
):
    response = api_client.get(
        "/conversations"
    )

    assert response.status_code == 401

    assert response.json()["error_code"] == (
        "authentication_required"
    )


def test_get_conversations_returns_empty_list(
    api_client,
):
    _register_and_login(
        api_client,
        "empty@example.com",
    )

    response = api_client.get(
        "/conversations"
    )

    assert response.status_code == 200

    assert response.json() == {
        "conversations": []
    }


def test_get_conversations_returns_saved_conversation(
    api_client,
):
    user = _register_and_login(
        api_client,
        "saved@example.com",
    )

    database.create_or_update_conversation(
        thread_id="thread-001",
        first_message="Hello RobBotGPT",
        user_id=user["id"],
    )

    response = api_client.get(
        "/conversations"
    )

    assert response.status_code == 200

    data = response.json()

    assert "conversations" in data
    assert len(data["conversations"]) == 1

    conversation = (
        data["conversations"][0]
    )

    assert (
        conversation["thread_id"]
        == "thread-001"
    )

    assert (
        conversation["title"]
        == "Hello RobBotGPT"
    )

    assert "created_at" in conversation
    assert "updated_at" in conversation


def test_get_conversations_returns_multiple_conversations(
    api_client,
):
    user = _register_and_login(
        api_client,
        "multiple@example.com",
    )

    database.create_or_update_conversation(
        thread_id="thread-A",
        first_message="Conversation A",
        user_id=user["id"],
    )

    database.create_or_update_conversation(
        thread_id="thread-B",
        first_message="Conversation B",
        user_id=user["id"],
    )

    response = api_client.get(
        "/conversations"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(
        data["conversations"]
    ) == 2

    thread_ids = {
        conversation["thread_id"]
        for conversation
        in data["conversations"]
    }

    assert thread_ids == {
        "thread-A",
        "thread-B",
    }


def test_get_conversations_is_isolated_by_user(
    api_client,
):
    user_a = _register_and_login(
        api_client,
        "user-a@example.com",
    )

    database.create_or_update_conversation(
        thread_id="thread-user-a",
        first_message="User A conversation",
        user_id=user_a["id"],
    )

    api_client.post(
        "/auth/logout"
    )

    user_b = _register_and_login(
        api_client,
        "user-b@example.com",
    )

    database.create_or_update_conversation(
        thread_id="thread-user-b",
        first_message="User B conversation",
        user_id=user_b["id"],
    )

    response = api_client.get(
        "/conversations"
    )

    assert response.status_code == 200

    conversations = (
        response.json()["conversations"]
    )

    assert len(conversations) == 1

    assert (
        conversations[0]["thread_id"]
        == "thread-user-b"
    )