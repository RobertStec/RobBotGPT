import pytest

import services.document_service as document_service

from core.exceptions import (
    DocumentProcessingError,
    UnsupportedFileTypeError,
)



def test_process_uploaded_document_success(
    tmp_path,
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        document_service.settings,
        "upload_dir",
        tmp_path,
    )

    mocker.patch(
        "services.document_service.uuid.uuid4",
        return_value="file-id",
    )

    mock_add = mocker.patch.object(
        document_service,
        "add_document_to_rag",
        return_value={
            "filename": "document.txt",
            "chunks": 3,
        },
    )

    mock_conversation = mocker.patch.object(
        document_service,
        "create_or_update_conversation",
    )

    result = (
        document_service
        .process_uploaded_document(
            filename="document.txt",
            content=b"Document content",
            thread_id="thread-1",
        )
    )

    assert result == {
        "filename": "document.txt",
        "chunks": 3,
    }

    expected_path = (
        tmp_path
        / "file-id_document.txt"
    )

    assert expected_path.read_bytes() == (
        b"Document content"
    )

    mock_add.assert_called_once_with(
        file_path=str(expected_path),
        thread_id="thread-1",
        original_filename="document.txt",
    )

    mock_conversation.assert_called_once_with(
        "thread-1",
        "Uploaded document",
    )



# Test unsupported

def test_process_uploaded_document_rejects_unsupported_extension(
    tmp_path,
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        document_service.settings,
        "upload_dir",
        tmp_path,
    )

    mock_add = mocker.patch.object(
        document_service,
        "add_document_to_rag",
    )

    mock_conversation = mocker.patch.object(
        document_service,
        "create_or_update_conversation",
    )

    with pytest.raises(
        UnsupportedFileTypeError
    ):
        document_service.process_uploaded_document(
            filename="image.jpg",
            content=b"image",
            thread_id="thread-1",
        )

    mock_add.assert_not_called()
    mock_conversation.assert_not_called()

    assert list(
        tmp_path.iterdir()
    ) == []




# Test awarii RAG

def test_process_uploaded_document_rolls_back_when_rag_fails(
    tmp_path,
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        document_service.settings,
        "upload_dir",
        tmp_path,
    )

    mocker.patch(
        "services.document_service.uuid.uuid4",
        return_value="file-id",
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

    mock_conversation = mocker.patch.object(
        document_service,
        "create_or_update_conversation",
    )

    with pytest.raises(
        DocumentProcessingError
    ):
        document_service.process_uploaded_document(
            filename="document.txt",
            content=b"content",
            thread_id="thread-error",
        )

    mock_cleanup.assert_called_once_with(
        thread_id="thread-error",
        stored_name="file-id_document.txt",
    )

    mock_conversation.assert_not_called()



# Test awarii DB po udanym RAG

def test_process_uploaded_document_rolls_back_when_database_fails(
    tmp_path,
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        document_service.settings,
        "upload_dir",
        tmp_path,
    )

    mocker.patch(
        "services.document_service.uuid.uuid4",
        return_value="file-id",
    )

    mocker.patch.object(
        document_service,
        "add_document_to_rag",
        return_value={
            "filename": "document.txt",
            "chunks": 2,
        },
    )

    mocker.patch.object(
        document_service,
        "create_or_update_conversation",
        side_effect=RuntimeError(
            "Database failed"
        ),
    )

    mock_cleanup = mocker.patch.object(
        document_service,
        "delete_document_from_rag",
        return_value={
            "deleted_chunks": 2,
            "deleted_files": 1,
        },
    )

    with pytest.raises(
        DocumentProcessingError
    ):
        document_service.process_uploaded_document(
            filename="document.txt",
            content=b"content",
            thread_id="thread-db-error",
        )

    mock_cleanup.assert_called_once_with(
        thread_id="thread-db-error",
        stored_name="file-id_document.txt",
    )



# Test fallback rollbacku

def test_process_uploaded_document_removes_file_when_rag_cleanup_fails(
    tmp_path,
    monkeypatch,
    mocker,
):
    monkeypatch.setattr(
        document_service.settings,
        "upload_dir",
        tmp_path,
    )

    mocker.patch(
        "services.document_service.uuid.uuid4",
        return_value="file-id",
    )

    mocker.patch.object(
        document_service,
        "add_document_to_rag",
        side_effect=RuntimeError(
            "RAG failed"
        ),
    )

    mocker.patch.object(
        document_service,
        "delete_document_from_rag",
        side_effect=RuntimeError(
            "Rollback failed"
        ),
    )

    expected_path = (
        tmp_path
        / "file-id_document.txt"
    )

    with pytest.raises(
        DocumentProcessingError
    ):
        document_service.process_uploaded_document(
            filename="document.txt",
            content=b"content",
            thread_id="thread-1",
        )

    assert (
        expected_path.exists()
        is False
    )