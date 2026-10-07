import database


def test_history_returns_empty_list_for_unknown_thread(
    api_client,
):
    response = api_client.get(
        "/history/unknown-thread"
    )

    assert response.status_code == 200

    assert response.json() == {
        "messages": []
    }


def test_history_returns_saved_messages(
    api_client,
):
    database.create_or_update_conversation(
        thread_id="thread-history",
        first_message="Hello",
    )

    database.save_chat_message(
        thread_id="thread-history",
        role="user",
        content="Hello RobBotGPT",
    )

    database.save_chat_message(
        thread_id="thread-history",
        role="assistant",
        content="Hello!",
    )

    response = api_client.get(
        "/history/thread-history"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["messages"] == [
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




# Historia odpowiedzi RAG ze źródłami

def test_history_deserializes_rag_sources(
    api_client,
):
    database.create_or_update_conversation(
        thread_id="thread-rag",
        first_message="Question about document",
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
    )

    response = api_client.get(
        "/history/thread-rag"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(
        data["messages"]
    ) == 1

    message = (
        data["messages"][0]
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



# Test izolacji thread_id przez API

def test_history_is_isolated_by_thread_id(
    api_client,
):
    database.create_or_update_conversation(
        "thread-A",
        "Conversation A",
    )

    database.create_or_update_conversation(
        "thread-B",
        "Conversation B",
    )

    database.save_chat_message(
        "thread-A",
        "user",
        "Message from A",
    )

    database.save_chat_message(
        "thread-B",
        "user",
        "Message from B",
    )

    response = api_client.get(
        "/history/thread-A"
    )

    assert response.status_code == 200

    messages = (
        response
        .json()["messages"]
    )

    assert len(messages) == 1

    assert (
        messages[0]["content"]
        == "Message from A"
    )

    contents = [
        message["content"]
        for message in messages
    ]

    assert (
        "Message from B"
        not in contents
    )