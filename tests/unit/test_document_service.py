import io
import zipfile
import pytest
import services.document_service as document_service

from pypdf import PdfWriter

from core.exceptions import (
    DocumentProcessingError,
    UnsupportedFileTypeError,
    InvalidUploadFilenameError,
    InvalidUploadContentError,
)


USER_ID = 101



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

    mock_exists = mocker.patch.object(
        document_service,
        "conversation_exists",
        return_value=False,
    )

    mock_conversation = mocker.patch.object(
        document_service,
        "create_or_update_conversation",
        return_value=True,
    )

    result = (
        document_service
        .process_uploaded_document(
            filename="document.txt",
            content=b"Document content",
            thread_id="thread-1",
            user_id=USER_ID,
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
        user_id=USER_ID,
    )

    mock_conversation.assert_called_once_with(
        "thread-1",
        "Uploaded document",
        user_id=USER_ID,
    )

    mock_exists.assert_called_once_with(
        "thread-1",
        user_id=USER_ID,
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
            user_id=USER_ID,
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

    mocker.patch.object(
        document_service,
        "conversation_exists",
        return_value=False,
    )

    mock_conversation = mocker.patch.object(
        document_service,
        "create_or_update_conversation",
        return_value=True,
    )

    mock_delete_conversation = mocker.patch.object(
        document_service,
        "delete_conversation",
        return_value=True,
    )

    with pytest.raises(
        DocumentProcessingError
    ):
        document_service.process_uploaded_document(
            filename="document.txt",
            content=b"content",
            thread_id="thread-error",
            user_id=USER_ID,
        )

    mock_cleanup.assert_called_once_with(
        thread_id="thread-error",
        stored_name="file-id_document.txt",
        user_id=USER_ID,
    )

    mock_conversation.assert_called_once_with(
        "thread-error",
        "Uploaded document",
        user_id=USER_ID,
    )

    mock_delete_conversation.assert_called_once_with(
        "thread-error",
        user_id=USER_ID,
)



# Test awarii DB po udanym RAG

def test_process_uploaded_document_stops_before_rag_when_database_fails(
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
        "conversation_exists",
        return_value=False,
    )

    mock_add = mocker.patch.object(
        document_service,
        "add_document_to_rag",
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
    )

    with pytest.raises(
        DocumentProcessingError
    ):
        document_service.process_uploaded_document(
            filename="document.txt",
            content=b"content",
            thread_id="thread-db-error",
            user_id=USER_ID,
        )

    mock_add.assert_not_called()
    mock_cleanup.assert_not_called()

    assert list(
        tmp_path.iterdir()
    ) == []



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

    mocker.patch.object(
        document_service,
        "conversation_exists",
        return_value=False,
    )

    mocker.patch.object(
        document_service,
        "create_or_update_conversation",
        return_value=True,
    )

    mock_delete_conversation = mocker.patch.object(
        document_service,
        "delete_conversation",
        return_value=True,
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
            user_id=USER_ID,
        )

    assert (
        expected_path.exists()
        is False
    )

    mock_delete_conversation.assert_called_once_with(
        "thread-1",
        user_id=USER_ID,
    )



@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
        "../secret.txt",
        "..\\secret.txt",
        "folder/file.txt",
        "folder\\file.txt",
        "bad\x00name.txt",
        "bad\nname.txt",
    ],
)
def test_validate_upload_filename_rejects_invalid_names(
    filename,
):
    with pytest.raises(
        InvalidUploadFilenameError
    ):
        document_service.validate_upload_filename(
            filename
        )



def test_validate_upload_filename_rejects_long_name():
    filename = (
        "a" * 181
        + ".txt"
    )

    with pytest.raises(
        InvalidUploadFilenameError
    ):
        document_service.validate_upload_filename(
            filename
        )




def test_validate_upload_filename_normalizes_spaces():
    result = (
        document_service
        .validate_upload_filename(
            "my document.txt"
        )
    )

    assert result == "my_document.txt"



def test_validate_upload_content_accepts_utf8_text():
    document_service.validate_upload_content(
        "notes.txt",
        "Zażółć gęślą jaźń".encode("utf-8"),
    )



def test_validate_upload_content_rejects_binary_text():
    with pytest.raises(
        InvalidUploadContentError
    ):
        document_service.validate_upload_content(
            "notes.txt",
            b"MZ\x00\x00\x90\x00binary",
        )


def test_validate_upload_content_accepts_real_pdf():
    buffer = io.BytesIO()

    writer = PdfWriter()
    writer.add_blank_page(
        width=72,
        height=72,
    )
    writer.write(buffer)

    document_service.validate_upload_content(
        "document.pdf",
        buffer.getvalue(),
    )



def test_validate_upload_content_rejects_fake_pdf():
    with pytest.raises(
        InvalidUploadContentError
    ):
        document_service.validate_upload_content(
            "document.pdf",
            b"This is not a PDF.",
        )



def test_validate_upload_content_accepts_docx_structure():
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
    ) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types/>",
        )
        archive.writestr(
            "_rels/.rels",
            "<Relationships/>",
        )
        archive.writestr(
            "word/document.xml",
            "<document/>",
        )

    document_service.validate_upload_content(
        "document.docx",
        buffer.getvalue(),
    )



def test_validate_upload_content_rejects_fake_docx():
    with pytest.raises(
        InvalidUploadContentError
    ):
        document_service.validate_upload_content(
            "document.docx",
            b"not a zip file",
        )



# Test traversal wewnątrz ZIP

def test_validate_docx_rejects_path_traversal():
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
    ) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types/>",
        )
        archive.writestr(
            "_rels/.rels",
            "<Relationships/>",
        )
        archive.writestr(
            "word/document.xml",
            "<document/>",
        )
        archive.writestr(
            "../evil.txt",
            "evil",
        )

    with pytest.raises(
        InvalidUploadContentError
    ):
        document_service.validate_upload_content(
            "document.docx",
            buffer.getvalue(),
        )



# Test liczby plików

def test_validate_docx_rejects_too_many_members(
    monkeypatch,
):
    monkeypatch.setattr(
        document_service,
        "MAX_DOCX_MEMBERS",
        3,
    )

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
    ) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types/>",
        )
        archive.writestr(
            "_rels/.rels",
            "<Relationships/>",
        )
        archive.writestr(
            "word/document.xml",
            "<document/>",
        )
        archive.writestr(
            "word/extra.xml",
            "<extra/>",
        )

    with pytest.raises(
        InvalidUploadContentError
    ):
        document_service.validate_upload_content(
            "document.docx",
            buffer.getvalue(),
        )



# Test rozmiaru po dekompresji

def test_validate_docx_rejects_large_uncompressed_size(
    monkeypatch,
):
    monkeypatch.setattr(
        document_service,
        "MAX_DOCX_UNCOMPRESSED_BYTES",
        100,
    )

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
    ) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types/>",
        )
        archive.writestr(
            "_rels/.rels",
            "<Relationships/>",
        )
        archive.writestr(
            "word/document.xml",
            "x" * 101,
        )

    with pytest.raises(
        InvalidUploadContentError
    ):
        document_service.validate_upload_content(
            "document.docx",
            buffer.getvalue(),
        )



# Test wysokiego współczynnika kompresji

def test_validate_docx_rejects_extreme_compression_ratio(
    monkeypatch,
):
    monkeypatch.setattr(
        document_service,
        "MAX_DOCX_COMPRESSION_RATIO",
        2.0,
    )

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types/>",
        )
        archive.writestr(
            "_rels/.rels",
            "<Relationships/>",
        )
        archive.writestr(
            "word/document.xml",
            "A" * 5000,
        )

    with pytest.raises(
        InvalidUploadContentError
    ):
        document_service.validate_upload_content(
            "document.docx",
            buffer.getvalue(),
        )