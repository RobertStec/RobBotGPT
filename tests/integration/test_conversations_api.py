import database


def test_get_conversations_returns_empty_list(
    api_client,
):
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
    database.create_or_update_conversation(
        thread_id="thread-001",
        first_message="Hello RobBotGPT",
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
    database.create_or_update_conversation(
        thread_id="thread-A",
        first_message="Conversation A",
    )

    database.create_or_update_conversation(
        thread_id="thread-B",
        first_message="Conversation B",
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