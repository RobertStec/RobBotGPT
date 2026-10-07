import database
import services.conversation_service as conversation_service


def test_delete_conversation_removes_database_data_and_calls_dependencies(
    api_client,
    mocker,
):
    # -----------------------------------------
    # Prepare conversation data
    # -----------------------------------------

    database.create_or_update_conversation(
        thread_id="thread-delete",
        first_message="Conversation to delete",
    )

    database.save_chat_message(
        thread_id="thread-delete",
        role="user",
        content="Hello",
    )

    database.save_memory(
        thread_id="thread-delete",
        memory="Test memory",
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
        "thread-delete"
    )

    mock_delete_checkpoints.assert_called_once_with(
        "thread-delete"
    )

    # -----------------------------------------
    # Database must be empty
    # -----------------------------------------

    assert (
        database.get_chat_history(
            "thread-delete"
        )
        == []
    )

    assert (
        database.search_memory(
            "thread-delete",
            "anything",
        )
        == "No saved memory found."
    )

    conversations = (
        database.list_conversations()
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
    database.create_or_update_conversation(
        thread_id="thread-error",
        first_message="Keep me",
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
        database.list_conversations()
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
    database.create_or_update_conversation(
        thread_id="thread-checkpoint-error",
        first_message="Keep me",
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
        "thread-checkpoint-error"
    )

    assert database.conversation_exists(
        "thread-checkpoint-error"
    )




# Test awarii SQL

def test_delete_conversation_keeps_database_when_final_database_delete_fails(
    api_client,
    mocker,
):
    thread_id = "thread-db-delete-error"

    database.create_or_update_conversation(
        thread_id=thread_id,
        first_message="Keep me",
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
        thread_id
    )

    mock_delete_checkpoints.assert_called_once_with(
        thread_id
    )

    assert database.conversation_exists(
        thread_id
    )




# Test retry

def test_delete_conversation_can_be_retried_after_partial_failure(
    api_client,
    mocker,
):
    thread_id = "thread-retry"

    database.create_or_update_conversation(
        thread_id=thread_id,
        first_message="Retry me",
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
        thread_id
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
            thread_id
        )
        is False
    )

    assert mock_delete_rag.call_count == 2

    assert (
        mock_delete_checkpoints.call_count
        == 2
    )