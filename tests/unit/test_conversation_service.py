from datetime import datetime

import pytest

import services.conversation_service as conversation_service

from core.exceptions import (
    ConversationDeletionError,
    ConversationNotFoundError,
)



# Test listy

def test_list_conversation_summaries(
    mocker,
):
    created_at = datetime(
        2026, 10, 1, 12, 0
    )

    updated_at = datetime(
        2026, 10, 2, 13, 30
    )

    item = mocker.Mock(
        thread_id="thread-1",
        title="Test conversation",
        created_at=created_at,
        updated_at=updated_at,
    )

    mocker.patch.object(
        conversation_service,
        "list_conversations",
        return_value=[item],
    )

    result = (
        conversation_service
        .list_conversation_summaries()
    )

    assert result == [
        {
            "thread_id": "thread-1",
            "title": "Test conversation",
            "created_at": (
                "2026-10-01T12:00:00"
            ),
            "updated_at": (
                "2026-10-02T13:30:00"
            ),
        }
    ]



# Test historii

def test_get_conversation_history(
    mocker,
):
    message = mocker.Mock(
        role="assistant",
        content="Answer",
        sources=(
            '[{"source": "manual.pdf"}]'
        ),
    )

    mocker.patch.object(
        conversation_service,
        "get_chat_history",
        return_value=[message],
    )

    result = (
        conversation_service
        .get_conversation_history(
            "thread-1"
        )
    )

    assert result == [
        {
            "role": "assistant",
            "content": "Answer",
            "sources": [
                {
                    "source": "manual.pdf"
                }
            ],
        }
    ]



# Test poprawnego DELETE

def test_delete_conversation_resources_success(
    mocker,
):
    mocker.patch.object(
        conversation_service,
        "conversation_exists",
        return_value=True,
    )

    mock_rag = mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        return_value={
            "deleted_chunks": 4,
            "deleted_files": 1,
        },
    )

    mock_checkpoints = (
        mocker.patch.object(
            conversation_service,
            "delete_thread_checkpoints",
        )
    )

    mock_db = mocker.patch.object(
        conversation_service,
        "delete_conversation_from_db",
        return_value=True,
    )

    result = (
        conversation_service
        .delete_conversation_resources(
            "thread-1"
        )
    )

    assert result == {
        "message": (
            "Conversation deleted."
        ),
        "deleted_chunks": 4,
        "deleted_files": 1,
    }

    mock_rag.assert_called_once_with(
        "thread-1"
    )

    mock_checkpoints.assert_called_once_with(
        "thread-1"
    )

    mock_db.assert_called_once_with(
        "thread-1"
    )




# Test nieistniejącej rozmowy

def test_delete_unknown_conversation_stops_before_cleanup(
    mocker,
):
    mocker.patch.object(
        conversation_service,
        "conversation_exists",
        return_value=False,
    )

    mock_rag = mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
    )

    mock_checkpoints = (
        mocker.patch.object(
            conversation_service,
            "delete_thread_checkpoints",
        )
    )

    mock_db = mocker.patch.object(
        conversation_service,
        "delete_conversation_from_db",
    )

    with pytest.raises(
        ConversationNotFoundError
    ):
        conversation_service \
            .delete_conversation_resources(
                "unknown-thread"
            )

    mock_rag.assert_not_called()
    mock_checkpoints.assert_not_called()
    mock_db.assert_not_called()




# Test awarii RAG

def test_delete_conversation_stops_when_rag_cleanup_fails(
    mocker,
):
    mocker.patch.object(
        conversation_service,
        "conversation_exists",
        return_value=True,
    )

    mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        side_effect=RuntimeError(
            "RAG failed"
        ),
    )

    mock_checkpoints = (
        mocker.patch.object(
            conversation_service,
            "delete_thread_checkpoints",
        )
    )

    mock_db = mocker.patch.object(
        conversation_service,
        "delete_conversation_from_db",
    )

    with pytest.raises(
        ConversationDeletionError
    ):
        conversation_service \
            .delete_conversation_resources(
                "thread-1"
            )

    mock_checkpoints.assert_not_called()
    mock_db.assert_not_called()




# Test awarii checkpointów

def test_delete_conversation_keeps_db_when_checkpoint_cleanup_fails(
    mocker,
):
    mocker.patch.object(
        conversation_service,
        "conversation_exists",
        return_value=True,
    )

    mocker.patch.object(
        conversation_service,
        "delete_thread_documents",
        return_value={
            "deleted_chunks": 2,
            "deleted_files": 1,
        },
    )

    mocker.patch.object(
        conversation_service,
        "delete_thread_checkpoints",
        side_effect=RuntimeError(
            "Checkpoint failed"
        ),
    )

    mock_db = mocker.patch.object(
        conversation_service,
        "delete_conversation_from_db",
    )

    with pytest.raises(
        ConversationDeletionError
    ):
        conversation_service \
            .delete_conversation_resources(
                "thread-1"
            )

    mock_db.assert_not_called()