import database
import services.conversation_service as conversation_service

from core.exceptions import (
    RateLimitExceededError,
)



PASSWORD = "StrongPassword123!"



def test_delete_conversation_requires_authentication(
    api_client,
):
    response = api_client.delete(
        "/conversations/thread-001"
    )

    assert response.status_code == 401

    assert response.json()["error_code"] == (
        "authentication_required"
    )


def test_delete_conversation_of_another_user_returns_404(
    api_client,
    mocker,
):
    owner = _register_and_login(
        api_client,
        "delete-owner@example.com",
    )

    database.create_or_update_conversation(
        thread_id="private-delete",
        first_message="Private conversation",
        user_id=owner["id"],
    )

    api_client.post(
        "/auth/logout"
    )

    _register_and_login(
        api_client,
        "delete-other@example.com",
    )

    mock_delete_rag = mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
    )

    mock_delete_checkpoints = mocker.patch.object(
        conversation_service,
        "delete_thread_checkpoints",
    )

    response = api_client.delete(
        "/conversations/private-delete"
    )

    assert response.status_code == 404

    assert response.json()["error_code"] == (
        "conversation_not_found"
    )

    mock_delete_rag.assert_not_called()
    mock_delete_checkpoints.assert_not_called()

    assert database.conversation_exists(
        "private-delete",
        user_id=owner["id"],
    ) is True



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




def test_delete_conversation_removes_database_data_and_calls_dependencies(
    api_client,
    mocker,
):
    # -----------------------------------------
    # Prepare conversation data
    # -----------------------------------------

    user = _register_and_login(
        api_client,
        "delete@example.com",
    )

    database.create_or_update_conversation(
        thread_id="thread-delete",
        first_message="Conversation to delete",
        user_id=user["id"],
    )

    database.save_chat_message(
        thread_id="thread-delete",
        role="user",
        content="Hello",
        user_id=user["id"],
    )

    database.save_memory(
        thread_id="thread-delete",
        memory="Test memory",
        user_id=user["id"],
    )

    # -----------------------------------------
    # Mock external subsystems
    # -----------------------------------------

    mock_delete_rag = mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        return_value={
            "deleted_chunks": 4,
            "deleted_files": 1,
        },
    )

    mock_delete_checkpoints = mocker.patch.object(
        conversation_service,
        "delete_thread_checkpoints",
    )

    # -----------------------------------------
    # HTTP request
    # -----------------------------------------

    response = api_client.delete(
        "/conversations/thread-delete"
    )

    # -----------------------------------------
    # Response
    # -----------------------------------------

    assert response.status_code == 200

    assert response.json() == {
        "success": True,
        "message": "Conversation deleted.",
        "deleted_chunks": 4,
        "deleted_files": 1,
    }

    # -----------------------------------------
    # External calls
    # -----------------------------------------

    mock_delete_rag.assert_called_once_with(
        "thread-delete",
        user_id=user["id"],
    )

    mock_delete_checkpoints.assert_called_once_with(
        "thread-delete",
        user_id=user["id"],
    )

    # -----------------------------------------
    # Database must be empty
    # -----------------------------------------

    assert (
        database.get_chat_history(
            "thread-delete",
            user_id=user["id"],
        )
        == []
    )

    assert (
        database.search_memory(
            "thread-delete",
            "anything",
            user_id=user["id"],
        )
        == "No saved memory found."
    )

    conversations = (
        database.list_conversations(
            user_id=user["id"]
        )
    )
    assert all(
        conversation.thread_id
        != "thread-delete"
        for conversation in conversations
    )




# Test usuwania nieistniejącej rozmowy

def test_delete_unknown_conversation_returns_404(
    api_client,
    mocker,
):

    user = _register_and_login(
        api_client,
        "delete@example.com",
    )

    mock_delete_rag = mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        return_value={
            "deleted_chunks": 0,
            "deleted_files": 0,
        },
    )

    mock_delete_checkpoints = mocker.patch.object(
        conversation_service,
        "delete_thread_checkpoints",
    )

    response = api_client.delete(
        "/conversations/unknown-thread"
    )

    assert response.status_code == 404

    assert response.json() == {
        "success": False,
        "message": "Conversation not found.",
        "error_code": (
            "conversation_not_found"
        ),
    }

    mock_delete_rag.assert_not_called()

    mock_delete_checkpoints.assert_not_called()




# Test błędu jednego z subsystemów

def test_delete_conversation_returns_500_when_rag_cleanup_fails(
    api_client,
    mocker,
):

    user = _register_and_login(
        api_client,
        "delete@example.com",
    )

    database.create_or_update_conversation(
        thread_id="thread-error",
        first_message="Keep me",
        user_id=user["id"],
    )

    mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        side_effect=RuntimeError(
            "RAG cleanup failed"
        ),
    )

    mock_delete_checkpoints = mocker.patch.object(
        conversation_service,
        "delete_thread_checkpoints",
    )

    response = api_client.delete(
        "/conversations/thread-error"
    )

    assert response.status_code == 500

    assert response.json() == {
        "success": False,
        "message": (
            "Could not completely "
            "delete conversation."
        ),
        "error_code": (
            "conversation_deletion_failed"
        ),
    }

    assert (
        "RAG cleanup failed"
        not in response.text
    )

    mock_delete_checkpoints.assert_not_called()

    conversations = (
        database.list_conversations(
            user_id=user["id"]
        )
    )

    assert any(
        conversation.thread_id
        == "thread-error"
        for conversation in conversations
    )




# Test awarii checkpointów

def test_delete_conversation_keeps_database_when_checkpoint_cleanup_fails(
    api_client,
    mocker,
):

    user = _register_and_login(
        api_client,
        "delete@example.com",
    )

    database.create_or_update_conversation(
        thread_id="thread-checkpoint-error",
        first_message="Keep me",
        user_id=user["id"],
    )

    mock_delete_rag = mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        return_value={
            "deleted_chunks": 3,
            "deleted_files": 1,
        },
    )

    mocker.patch.object(
        conversation_service,
        "delete_thread_checkpoints",
        side_effect=RuntimeError(
            "Checkpoint database failed"
        ),
    )

    response = api_client.delete(
        "/conversations/"
        "thread-checkpoint-error"
    )

    assert response.status_code == 500

    assert response.json() == {
        "success": False,
        "message": (
            "Could not completely "
            "delete conversation."
        ),
        "error_code": (
            "conversation_deletion_failed"
        ),
    }

    assert (
        "Checkpoint database failed"
        not in response.text
    )

    mock_delete_rag.assert_called_once_with(
        "thread-checkpoint-error",
        user_id=user["id"],
    )

    assert database.conversation_exists(
        "thread-checkpoint-error",
        user_id=user["id"],
    )




# Test awarii SQL

def test_delete_conversation_keeps_database_when_final_database_delete_fails(
    api_client,
    mocker,
):

    user = _register_and_login(
        api_client,
        "delete@example.com",
    )

    thread_id = "thread-db-delete-error"

    database.create_or_update_conversation(
        thread_id=thread_id,
        first_message="Keep me",
        user_id=user["id"],
    )

    mock_delete_rag = mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        return_value={
            "deleted_chunks": 2,
            "deleted_files": 1,
        },
    )

    mock_delete_checkpoints = (
        mocker.patch.object(
            conversation_service,
            "delete_thread_checkpoints",
        )
    )

    mocker.patch.object(
        conversation_service,
        "delete_conversation_from_db",
        side_effect=RuntimeError(
            "Database delete failed"
        ),
    )

    response = api_client.delete(
        f"/conversations/{thread_id}"
    )

    assert response.status_code == 500

    assert response.json() == {
        "success": False,
        "message": (
            "Could not completely "
            "delete conversation."
        ),
        "error_code": (
            "conversation_deletion_failed"
        ),
    }

    assert (
        "Database delete failed"
        not in response.text
    )

    mock_delete_rag.assert_called_once_with(
        thread_id,
        user_id=user["id"],
    )

    mock_delete_checkpoints.assert_called_once_with(
        thread_id,
        user_id=user["id"],
    )

    assert database.conversation_exists(
        thread_id,
        user_id=user["id"],
    )




# Test retry

def test_delete_conversation_can_be_retried_after_partial_failure(
    api_client,
    mocker,
):
    user = _register_and_login(
        api_client,
        "delete@example.com",
    )

    thread_id = "thread-retry"

    database.create_or_update_conversation(
        thread_id=thread_id,
        first_message="Retry me",
        user_id=user["id"],
    )

    mock_delete_rag = mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        side_effect=[
            {
                "deleted_chunks": 4,
                "deleted_files": 1,
            },
            {
                "deleted_chunks": 0,
                "deleted_files": 0,
            },
        ],
    )

    mock_delete_checkpoints = (
        mocker.patch.object(
            conversation_service,
            "delete_thread_checkpoints",
            side_effect=[
                RuntimeError(
                    "Temporary checkpoint error"
                ),
                None,
            ],
        )
    )

    first_response = api_client.delete(
        f"/conversations/{thread_id}"
    )

    assert first_response.status_code == 500

    assert database.conversation_exists(
        thread_id,
        user_id=user["id"],
    )

    second_response = api_client.delete(
        f"/conversations/{thread_id}"
    )

    assert second_response.status_code == 200

    assert second_response.json() == {
        "success": True,
        "message": "Conversation deleted.",
        "deleted_chunks": 0,
        "deleted_files": 0,
    }

    assert (
        database.conversation_exists(
            thread_id,
            user_id=user["id"],
        )
        is False
    )

    assert mock_delete_rag.call_count == 2

    assert (
        mock_delete_checkpoints.call_count
        == 2
    )



def test_delete_conversation_applies_user_rate_limit(
    api_client,
    mocker,
):
    user = _register_and_login(
        api_client,
        "delete-rate@example.com",
    )

    database.create_or_update_conversation(
        thread_id="delete-rate-thread",
        first_message="Delete test",
        user_id=user["id"],
    )

    mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        return_value={
            "deleted_chunks": 0,
            "deleted_files": 0,
        },
    )

    mocker.patch.object(
        conversation_service,
        "delete_thread_checkpoints",
    )

    mock_enforce = mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new_callable=mocker.AsyncMock,
    )

    response = api_client.delete(
        "/conversations/delete-rate-thread"
    )

    assert response.status_code == 200

    mock_enforce.assert_awaited_once_with(
        scope="conversation_delete",
        identifier=f"user:{user['id']}",
        limit=20,
        window_seconds=60,
    )




def test_delete_conversation_returns_429_when_rate_limit_exceeded(
    api_client,
    mocker,
):
    user = _register_and_login(
        api_client,
        "delete-blocked@example.com",
    )

    database.create_or_update_conversation(
        thread_id="delete-blocked-thread",
        first_message="Keep me",
        user_id=user["id"],
    )

    mock_delete_rag = mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
    )

    mock_delete_checkpoints = (
        mocker.patch.object(
            conversation_service,
            "delete_thread_checkpoints",
        )
    )

    mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new=mocker.AsyncMock(
            side_effect=RateLimitExceededError(
                retry_after=18
            )
        ),
    )

    response = api_client.delete(
        "/conversations/delete-blocked-thread"
    )

    assert response.status_code == 429

    assert response.headers[
        "retry-after"
    ] == "18"

    assert response.json()[
        "error_code"
    ] == "rate_limit_exceeded"

    mock_delete_rag.assert_not_called()
    mock_delete_checkpoints.assert_not_called()

    assert database.conversation_exists(
        "delete-blocked-thread",
        user_id=user["id"],
    ) is True




def test_delete_rejects_invalid_thread_id(
    api_client,
):
    _register_and_login(
        api_client,
        "delete-invalid-thread@example.com",
    )

    response = api_client.delete(
        "/conversations/bad%20thread"
    )

    assert response.status_code == 422

    assert response.json() == {
        "success": False,
        "message": "Request validation failed.",
        "error_code": "request_validation_failed",
    }