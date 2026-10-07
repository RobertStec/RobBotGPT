import database
from pathlib import Path
import services.document_service as document_service


# Poprawny upload TXT

def test_upload_document_success(
    api_client,
    mocker,
):
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
        database.list_conversations()
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
    captured = {}

    def fake_add_document_to_rag(
        file_path,
        thread_id,
        original_filename,
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
        captured["filename"]
        == "notes.txt"
    )




# Test nieobsługiwany format

def test_upload_rejects_unsupported_file_type(
    api_client,
    mocker,
):
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



# test błąd RAG podczas uploadu

def test_upload_returns_500_when_rag_ingestion_fails(
    api_client,
    mocker,
):
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
        cleanup_kwargs["stored_name"]
        .endswith("_document.txt")
    )

    conversations = (
        database.list_conversations()
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




# Test Brak pliku

def test_upload_requires_file(
    api_client,
):
    response = api_client.post(
        "/upload",
        data={
            "thread_id": "thread-123"
        },
    )

    assert response.status_code == 422



# Test DB pada po udanym RAG

def test_upload_rolls_back_when_conversation_save_fails(
    api_client,
    mocker,
):
    mocker.patch.object(
        document_service,
        "add_document_to_rag",
        return_value={
            "filename": "document.txt",
            "chunks": 3,
        },
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
        return_value={
            "deleted_chunks": 3,
            "deleted_files": 1,
        },
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

    mock_cleanup.assert_called_once()

    cleanup_kwargs = (
        mock_cleanup
        .call_args
        .kwargs
    )

    assert (
        cleanup_kwargs["thread_id"]
        == "thread-db-error"
    )

    assert (
        cleanup_kwargs["stored_name"]
        .endswith("_document.txt")
    )
