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



# Test potwierdza przepływ HTTP → SQLAlchemy → PostgreSQL.

def test_postgres_api_lists_conversations(
    postgres_api_client,
    postgres_database,
):
    user = _register_and_login(
        postgres_api_client,
        "postgres-api-list@example.com",
    )

    database.create_or_update_conversation(
        "postgres-api-list",
        "PostgreSQL API test",
        user_id=user["id"],
    )

    response = postgres_api_client.get(
        "/conversations"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(
        data["conversations"]
    ) == 1

    conversation = (
        data["conversations"][0]
    )

    assert (
        conversation["thread_id"]
        == "postgres-api-list"
    )

    assert (
        conversation["title"]
        == "PostgreSQL API test"
    )




# Test endpoint historii deserializuje zapisane sources i zwraca je do frontendu jako listę

def test_postgres_api_returns_chat_history(
    postgres_api_client,
    postgres_database,
):
    user = _register_and_login(
        postgres_api_client,
        "postgres-api-history@example.com",
    )

    thread_id = (
        "postgres-api-history"
    )

    database.create_or_update_conversation(
        thread_id,
        "History test",
        user_id=user["id"],
    )

    database.save_chat_message(
        thread_id=thread_id,
        role="user",
        content="Hello PostgreSQL",
        user_id=user["id"],
    )

    sources = [
        {
            "source": "manual.pdf",
            "page": 2,
            "chunk_index": 4,
        }
    ]

    database.save_chat_message(
        thread_id=thread_id,
        role="assistant",
        content="PostgreSQL response",
        sources=sources,
        user_id=user["id"],
    )

    response = postgres_api_client.get(
        f"/history/{thread_id}"
    )

    assert response.status_code == 200

    assert response.json() == {
        "messages": [
            {
                "role": "user",
                "content": "Hello PostgreSQL",
                "sources": [],
            },
            {
                "role": "assistant",
                "content": "PostgreSQL response",
                "sources": sources,
            },
        ]
    }



# Test zapis przez API

def test_postgres_upload_endpoint_creates_conversation(
    postgres_api_client,
    postgres_database,
):
    user = _register_and_login(
        postgres_api_client,
        "postgres-api-upload@example.com",
    )

    thread_id = (
        "postgres-api-upload"
    )

    response = postgres_api_client.post(
        "/upload",
        data={
            "thread_id": thread_id,
        },
        files={
            "file": (
                "test.txt",
                b"PostgreSQL API integration test.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["filename"] == "test.txt"
    assert data["chunks"] == 1

    assert (
        database.conversation_exists(
            thread_id,
            user_id=user["id"],
        )
        is True
    )

    db = postgres_database()

    try:
        conversation = (
            db.query(
                database.Conversation
            )
            .filter(
                database.Conversation.thread_id
                == thread_id,
                database.Conversation.user_id
                == user["id"],
            )
            .one()
        )

        assert (
            conversation.title
            == "Uploaded document"
        )

        assert conversation.user_id == user["id"]

    finally:
        db.close()