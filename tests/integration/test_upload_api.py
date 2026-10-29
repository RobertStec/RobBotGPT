import pytest
import database
from pathlib import Path

import core.rate_limit as rate_limit
import services.document_service as document_service

from core.exceptions import (
    RateLimitExceededError,
)



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


def test_upload_requires_authentication(
    api_client,
):
    response = api_client.post(
        "/upload",
        data={
            "thread_id": "thread-auth"
        },
        files={
            "file": (
                "document.txt",
                b"content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 401

    assert response.json()["error_code"] == (
        "authentication_required"
    )


# Poprawny upload TXT

def test_upload_document_success(
    api_client,
    mocker,
):
    user = _register_and_login(
        api_client,
        "upload-success@example.com",
    )

    mock_add_document = mocker.patch.object(
        document_service,
        "add_document_to_rag",
        return_value={
            "filename": "document.txt",
            "chunks": 3,
        },
    )

    response = api_client.post(
        "/upload",
        data={
            "thread_id": "thread-upload"
        },
        files={
            "file": (
                "document.txt",
                b"RobBotGPT uses LangGraph.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "success": True,
        "filename": "document.txt",
        "chunks": 3,
    }

    mock_add_document.assert_called_once()

    call_kwargs = (
        mock_add_document
        .call_args
        .kwargs
    )

    assert (
        call_kwargs["thread_id"]
        == "thread-upload"
    )

    assert (
        call_kwargs["user_id"]
        == user["id"]
    )

    assert (
        call_kwargs["original_filename"]
        == "document.txt"
    )

    saved_path = Path(
        call_kwargs["file_path"]
    )

    assert (
        saved_path.parent
        == Path("uploads")
    )

    assert (
        saved_path.name
        .endswith("_document.txt")
    )

    assert (
        saved_path.name
        != "document.txt"
    )

    assert (
        call_kwargs["file_path"]
        .endswith("_document.txt")
    )

    conversations = (
        database.list_conversations(
            user_id=user["id"]
        )
    )

    assert any(
        conversation.thread_id
        == "thread-upload"
        for conversation in conversations
    )


# Test czy plik naprawdę został zapisany

def test_upload_document_writes_file_before_rag_ingestion(
    api_client,
    mocker,
):
    user = _register_and_login(
        api_client,
        "upload-file@example.com",
    )

    captured = {}

    def fake_add_document_to_rag(
        file_path,
        thread_id,
        original_filename,
        *,
        user_id,
    ):
        with open(
            file_path,
            "rb",
        ) as file:
            captured["content"] = (
                file.read()
            )

        captured["thread_id"] = (
            thread_id
        )

        captured["user_id"] = (
            user_id
        )

        captured["filename"] = (
            original_filename
        )

        return {
            "chunks": 1
        }

    mocker.patch.object(
        document_service,
        "add_document_to_rag",
        side_effect=fake_add_document_to_rag,
    )

    response = api_client.post(
        "/upload",
        data={
            "thread_id": "thread-file"
        },
        files={
            "file": (
                "notes.txt",
                b"Important document content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    assert (
        captured["content"]
        == b"Important document content"
    )

    assert (
        captured["thread_id"]
        == "thread-file"
    )

    assert (
        captured["user_id"]
        == user["id"]
    )

    assert (
        captured["filename"]
        == "notes.txt"
    )


# Test nieobsługiwany format

def test_upload_rejects_unsupported_file_type(
    api_client,
    mocker,
):
    _register_and_login(
        api_client,
        "upload-invalid@example.com",
    )

    mock_add_document = mocker.patch.object(
        document_service,
        "add_document_to_rag",
    )

    response = api_client.post(
        "/upload",
        data={
            "thread_id": "thread-upload"
        },
        files={
            "file": (
                "image.jpg",
                b"fake image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "success": False,
        "message": (
            "Unsupported file type. "
            "Upload PDF, DOCX, TXT, MD, PY, or CSV."
        ),
        "error_code": "unsupported_file_type",
    }

    mock_add_document.assert_not_called()


# Test błąd RAG podczas uploadu

def test_upload_returns_500_when_rag_ingestion_fails(
    api_client,
    mocker,
):
    user = _register_and_login(
        api_client,
        "upload-rag-error@example.com",
    )

    mocker.patch.object(
        document_service,
        "add_document_to_rag",
        side_effect=RuntimeError(
            "Embedding failed"
        ),
    )

    mock_cleanup = mocker.patch.object(
        document_service,
        "delete_document_from_rag",
        return_value={
            "deleted_chunks": 0,
            "deleted_files": 1,
        },
    )

    response = api_client.post(
        "/upload",
        data={
            "thread_id": "thread-error"
        },
        files={
            "file": (
                "document.txt",
                b"Example document",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "success": False,
        "message": (
            "Could not process "
            "the uploaded document."
        ),
        "error_code": (
            "document_processing_failed"
        ),
    }

    assert (
        "Embedding failed"
        not in response.text
    )

    mock_cleanup.assert_called_once()

    cleanup_kwargs = (
        mock_cleanup
        .call_args
        .kwargs
    )

    assert (
        cleanup_kwargs["thread_id"]
        == "thread-error"
    )

    assert (
        cleanup_kwargs["user_id"]
        == user["id"]
    )

    assert (
        cleanup_kwargs["stored_name"]
        .endswith("_document.txt")
    )

    conversations = (
        database.list_conversations(
            user_id=user["id"]
        )
    )

    assert all(
        conversation.thread_id
        != "thread-error"
        for conversation in conversations
    )


# Test brak thread_id

def test_upload_requires_thread_id(
    api_client,
):
    _register_and_login(
        api_client,
        "upload-no-thread@example.com",
    )

    response = api_client.post(
        "/upload",
        files={
            "file": (
                "document.txt",
                b"content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422


# Test brak pliku

def test_upload_requires_file(
    api_client,
):
    _register_and_login(
        api_client,
        "upload-no-file@example.com",
    )

    response = api_client.post(
        "/upload",
        data={
            "thread_id": "thread-123"
        },
    )

    assert response.status_code == 422


def test_upload_rejects_foreign_thread(
    api_client,
    mocker,
):
    owner = _register_and_login(
        api_client,
        "upload-owner@example.com",
    )

    database.create_or_update_conversation(
        thread_id="private-upload-thread",
        first_message="Private conversation",
        user_id=owner["id"],
    )

    api_client.post(
        "/auth/logout"
    )

    _register_and_login(
        api_client,
        "upload-other@example.com",
    )

    mock_add_document = mocker.patch.object(
        document_service,
        "add_document_to_rag",
    )

    response = api_client.post(
        "/upload",
        data={
            "thread_id": "private-upload-thread"
        },
        files={
            "file": (
                "secret.txt",
                b"Unauthorized upload",
                "text/plain",
            )
        },
    )

    assert response.status_code == 404

    assert response.json()["error_code"] == (
        "conversation_not_found"
    )

    mock_add_document.assert_not_called()


# Test DB pada przed RAG

def test_upload_returns_500_when_conversation_save_fails(
    api_client,
    mocker,
):
    _register_and_login(
        api_client,
        "upload-db-error@example.com",
    )

    mock_add_document = mocker.patch.object(
        document_service,
        "add_document_to_rag",
    )

    mocker.patch.object(
        document_service,
        "create_or_update_conversation",
        side_effect=RuntimeError(
            "Database write failed"
        ),
    )

    mock_cleanup = mocker.patch.object(
        document_service,
        "delete_document_from_rag",
    )

    response = api_client.post(
        "/upload",
        data={
            "thread_id": (
                "thread-db-error"
            )
        },
        files={
            "file": (
                "document.txt",
                b"Example document",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "success": False,
        "message": (
            "Could not process "
            "the uploaded document."
        ),
        "error_code": (
            "document_processing_failed"
        ),
    }

    assert (
        "Database write failed"
        not in response.text
    )

    mock_add_document.assert_not_called()
    mock_cleanup.assert_not_called()





def test_upload_applies_user_rate_limit(
    api_client,
    mocker,
):
    user = _register_and_login(
        api_client,
        "upload-rate-limit@example.com",
    )

    mock_enforce = mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new_callable=mocker.AsyncMock,
    )

    mocker.patch.object(
        document_service,
        "add_document_to_rag",
        return_value={
            "chunks": 1,
        },
    )

    response = api_client.post(
        "/upload",
        data={
            "thread_id": "rate-limit-upload"
        },
        files={
            "file": (
                "document.txt",
                b"Rate limit test",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    mock_enforce.assert_awaited_once_with(
        scope="document_upload",
        identifier=f"user:{user['id']}",
        limit=10,
        window_seconds=600,
    )




def test_upload_returns_429_when_rate_limit_exceeded(
    api_client,
    mocker,
):
    user = _register_and_login(
        api_client,
        "upload-rate-blocked@example.com",
    )

    mock_add_document = mocker.patch.object(
        document_service,
        "add_document_to_rag",
    )

    mocker.patch(
        "core.rate_limit.enforce_rate_limit",
        new=mocker.AsyncMock(
            side_effect=RateLimitExceededError(
                retry_after=180
            )
        ),
    )

    response = api_client.post(
        "/upload",
        data={
            "thread_id": "blocked-upload"
        },
        files={
            "file": (
                "document.txt",
                b"Blocked document",
                "text/plain",
            )
        },
    )

    assert response.status_code == 429

    assert (
        response.headers["retry-after"]
        == "180"
    )

    assert response.json() == {
        "success": False,
        "message": (
            "Too many requests. "
            "Please try again later."
        ),
        "error_code": "rate_limit_exceeded",
    }

    mock_add_document.assert_not_called()

    assert database.conversation_exists(
        "blocked-upload",
        user_id=user["id"],
    ) is False