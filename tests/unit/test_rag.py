import pytest

import rag





# Test odczytu zwykłych plików tekstowych

@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".txt", "Example TXT content"),
        (".md", "# Markdown content"),
        (".py", "print('hello')"),
        (".csv", "name,value\nA,1"),
    ],
)
def test_read_file_text_reads_text_files(
    tmp_path,
    suffix,
    content,
):
    file_path = tmp_path / f"example{suffix}"

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    result = rag.read_file_text(
        str(file_path)
    )

    assert result == content



@pytest.fixture
def mock_pgvector_storage(
    mocker,
):
    embeddings = mocker.Mock()

    embeddings.embed_documents.side_effect = (
        lambda texts: [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]
    )

    mocker.patch.object(
        rag,
        "get_embeddings",
        return_value=embeddings,
    )

    db = mocker.Mock()

    def assign_document_id():
        document = (
            db.add
            .call_args
            .args[0]
        )

        document.id = 123

    db.flush.side_effect = (
        assign_document_id
    )

    mocker.patch.object(
        rag,
        "create_session",
        return_value=db,
    )

    return embeddings, db



@pytest.fixture
def mock_pgvector_retrieval(
    mocker,
):
    embeddings = mocker.Mock()

    embeddings.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    mocker.patch.object(
        rag,
        "get_embeddings",
        return_value=embeddings,
    )

    db = mocker.Mock()

    mocker.patch.object(
        rag,
        "create_session",
        return_value=db,
    )

    return embeddings, db



@pytest.fixture
def mock_pgvector_delete(
    mocker,
):
    db = mocker.Mock()

    mocker.patch.object(
        rag,
        "create_session",
        return_value=db,
    )

    return db



# Nieobsługiwany format

def test_read_file_text_rejects_unsupported_file_type(
    tmp_path,
):
    file_path = tmp_path / "image.jpg"

    file_path.write_text(
        "fake image",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported file type",
    ):
        rag.read_file_text(
            str(file_path)
        )




# Test ingestowania dokumentu TXT

def test_add_document_to_rag_creates_chunks_with_metadata(
    tmp_path,
    mock_pgvector_storage,
):
    file_path = (
        tmp_path
        / "stored_document.txt"
    )

    file_path.write_text(
        "RobBotGPT uses LangGraph and LangChain.",
        encoding="utf-8",
    )

    embeddings, db = (
        mock_pgvector_storage
    )

    result = rag.add_document_to_rag(
        file_path=str(file_path),
        thread_id="thread-123",
        original_filename="project.txt",
    )

    assert result == {
        "filename": "project.txt",
        "chunks": 1,
    }

    embeddings.embed_documents.assert_called_once_with(
        [
            "RobBotGPT uses LangGraph and LangChain."
        ]
    )

    stored_document = (
        db.add.call_args.args[0]
    )

    assert isinstance(
        stored_document,
        rag.RagDocument,
    )

    assert (
        stored_document.thread_id
        == "thread-123"
    )

    assert (
        stored_document.source_name
        == "project.txt"
    )

    assert (
        stored_document.stored_name
        == "stored_document.txt"
    )

    db.flush.assert_called_once_with()

    stored_chunks = (
        db.add_all.call_args.args[0]
    )

    assert len(stored_chunks) == 1

    chunk = stored_chunks[0]

    assert isinstance(
        chunk,
        rag.RagChunk,
    )

    assert chunk.document_id == 123
    assert chunk.page is None
    assert chunk.chunk_index == 1

    assert (
        chunk.content
        == "RobBotGPT uses LangGraph and LangChain."
    )

    assert chunk.embedding == [
        0.1,
        0.2,
        0.3,
    ]

    db.commit.assert_called_once_with()
    db.rollback.assert_not_called()
    db.close.assert_called_once_with()




# Test chunkingu

def test_add_document_to_rag_splits_long_document(
    tmp_path,
    mock_pgvector_storage,
):
    file_path = tmp_path / "long.txt"

    content = (
        "LangGraph is used for agent workflows. "
        * 100
    )

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    result = rag.add_document_to_rag(
        file_path=str(file_path),
        thread_id="thread-long",
    )

    _, db = mock_pgvector_storage

    chunks = (
        db.add_all
        .call_args
        .args[0]
    )

    assert result["chunks"] > 1
    assert len(chunks) > 1

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == list(
        range(
            1,
            len(chunks) + 1,
        )
    )




# Pusty dokument

def test_add_document_to_rag_rejects_empty_document(
    tmp_path,
    mock_pgvector_storage,
):
    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "   \n   ",
        encoding="utf-8",
    )

    embeddings, db = (
        mock_pgvector_storage
    )

    with pytest.raises(
        ValueError,
        match="No text could be extracted",
    ):
        rag.add_document_to_rag(
            file_path=str(file_path),
            thread_id="thread-empty",
        )

    embeddings.embed_documents.assert_not_called()
    db.add.assert_not_called()




# Test zachowanie numerów stron PDF

def test_add_pdf_preserves_page_numbers(
    tmp_path,
    mock_pgvector_storage,
    mocker,
):
    file_path = (
        tmp_path
        / "stored_manual.pdf"
    )

    file_path.write_bytes(
        b"fake pdf"
    )

    _, db = mock_pgvector_storage

    page_1 = mocker.Mock()
    page_1.extract_text.return_value = (
        "Content from page one."
    )

    page_2 = mocker.Mock()
    page_2.extract_text.return_value = (
        "Content from page two."
    )

    fake_reader = mocker.Mock()

    fake_reader.pages = [
        page_1,
        page_2,
    ]

    mocker.patch(
        "rag.PdfReader",
        return_value=fake_reader,
    )

    result = rag.add_document_to_rag(
        file_path=str(file_path),
        thread_id="thread-pdf",
        original_filename="manual.pdf",
    )

    chunks = (
        db.add_all
        .call_args
        .args[0]
    )

    assert result == {
        "filename": "manual.pdf",
        "chunks": 2,
    }

    assert len(chunks) == 2

    assert chunks[0].page == 1
    assert chunks[1].page == 2

    assert chunks[0].chunk_index == 1
    assert chunks[1].chunk_index == 2

    assert (
        chunks[0].content
        == "Content from page one."
    )

    assert (
        chunks[1].content
        == "Content from page two."
    )





# Test Retrieval - poprawny filtr thread_id

def test_retrieve_from_rag_filters_by_thread_id(
    mock_pgvector_retrieval,
):
    embeddings, db = (
        mock_pgvector_retrieval
    )

    document = rag.RagDocument(
        id=10,
        thread_id="thread-123",
        source_name="project.pdf",
        stored_name="stored_project.pdf",
    )

    chunk = rag.RagChunk(
        id=20,
        document_id=10,
        page=2,
        chunk_index=3,
        content=(
            "LangGraph manages agent workflows."
        ),
        embedding=[
            0.1,
            0.2,
            0.3,
        ],
    )

    query = db.query.return_value

    query.join.return_value = query
    query.filter.return_value = query
    query.order_by.return_value = query
    query.limit.return_value = query

    query.all.return_value = [
        (
            chunk,
            document,
        )
    ]

    result = rag.retrieve_from_rag(
        query="What does LangGraph do?",
        thread_id="thread-123",
        k=4,
    )

    embeddings.embed_query.assert_called_once_with(
        "What does LangGraph do?"
    )

    db.query.assert_called_once_with(
        rag.RagChunk,
        rag.RagDocument,
    )

    query.limit.assert_called_once_with(
        4
    )

    assert (
        "LangGraph manages agent workflows."
        in result["context"]
    )

    assert (
        "[Source: project.pdf, page: 2]"
        in result["context"]
    )

    assert result["sources"] == [
        {
            "source": "project.pdf",
            "page": 2,
            "chunk_index": 3,
        }
    ]

    db.close.assert_called_once_with()



# Test Retrieval bez wyników

def test_retrieve_from_rag_returns_empty_result_when_no_documents(
    mock_pgvector_retrieval,
):
    embeddings, db = (
        mock_pgvector_retrieval
    )

    query = db.query.return_value

    query.join.return_value = query
    query.filter.return_value = query
    query.order_by.return_value = query
    query.limit.return_value = query

    query.all.return_value = []

    result = rag.retrieve_from_rag(
        query="Unknown information",
        thread_id="thread-empty",
        k=4,
    )

    assert result == {
        "context": "",
        "sources": [],
    }

    embeddings.embed_query.assert_called_once_with(
        "Unknown information"
    )

    db.close.assert_called_once_with()



# Test deduplikacji źródeł

def test_retrieve_from_rag_deduplicates_sources(
    mock_pgvector_retrieval,
):
    _, db = (
        mock_pgvector_retrieval
    )

    document = rag.RagDocument(
        id=1,
        thread_id="thread-123",
        source_name="manual.pdf",
        stored_name="stored_manual.pdf",
    )

    chunk_1 = rag.RagChunk(
        document_id=1,
        page=5,
        chunk_index=1,
        content="First chunk",
        embedding=[0.1, 0.2, 0.3],
    )

    chunk_2 = rag.RagChunk(
        document_id=1,
        page=5,
        chunk_index=2,
        content="Second chunk",
        embedding=[0.1, 0.2, 0.3],
    )

    query = db.query.return_value

    query.join.return_value = query
    query.filter.return_value = query
    query.order_by.return_value = query
    query.limit.return_value = query

    query.all.return_value = [
        (chunk_1, document),
        (chunk_2, document),
    ]

    result = rag.retrieve_from_rag(
        query="question",
        thread_id="thread-123",
    )

    assert len(
        result["sources"]
    ) == 1

    assert (
        result["sources"][0]["source"]
        == "manual.pdf"
    )

    assert (
        result["sources"][0]["page"]
        == 5
    )

    assert "First chunk" in result["context"]
    assert "Second chunk" in result["context"]




# Test Usuwania dokumentów dla konkretnego thread_id

def test_delete_thread_documents_removes_chunks_and_files(
    tmp_path,
    monkeypatch,
    mock_pgvector_delete,
    mocker,
):
    upload_dir = (
        tmp_path
        / "uploads"
    )

    upload_dir.mkdir()

    monkeypatch.setattr(
        rag.settings,
        "upload_dir",
        upload_dir,
    )

    file_1 = (
        upload_dir
        / "stored_1.pdf"
    )

    file_2 = (
        upload_dir
        / "stored_2.txt"
    )

    file_1.write_bytes(b"pdf")

    file_2.write_text(
        "text",
        encoding="utf-8",
    )

    document_1 = rag.RagDocument(
        id=1,
        thread_id="thread-delete",
        source_name="original.pdf",
        stored_name="stored_1.pdf",
    )

    document_2 = rag.RagDocument(
        id=2,
        thread_id="thread-delete",
        source_name="original.txt",
        stored_name="stored_2.txt",
    )

    db = mock_pgvector_delete

    document_query = mocker.Mock()
    chunk_query = mocker.Mock()

    db.query.side_effect = [
        document_query,
        chunk_query,
    ]

    (
        document_query
        .filter
        .return_value
        .all
        .return_value
    ) = [
        document_1,
        document_2,
    ]

    (
        chunk_query
        .filter
        .return_value
        .count
        .return_value
    ) = 3

    result = (
        rag.delete_thread_documents(
            "thread-delete"
        )
    )

    assert file_1.exists() is False
    assert file_2.exists() is False

    assert db.delete.call_count == 2

    deleted_documents = [
        call.args[0]
        for call
        in db.delete.call_args_list
    ]

    assert deleted_documents == [
        document_1,
        document_2,
    ]

    db.commit.assert_called_once_with()
    db.rollback.assert_not_called()
    db.close.assert_called_once_with()

    assert result == {
        "deleted_chunks": 3,
        "deleted_files": 2,
    }




# Brak dokumentów do usunięcia

def test_delete_thread_documents_handles_empty_thread(
    mock_pgvector_delete,
    mocker,
):
    db = mock_pgvector_delete

    document_query = mocker.Mock()

    db.query.return_value = (
        document_query
    )

    (
        document_query
        .filter
        .return_value
        .all
        .return_value
    ) = []

    result = (
        rag.delete_thread_documents(
            "unknown-thread"
        )
    )

    assert result == {
        "deleted_chunks": 0,
        "deleted_files": 0,
    }

    db.delete.assert_not_called()
    db.commit.assert_not_called()
    db.close.assert_called_once_with()



# =========================================================
# Additional file reader coverage
# =========================================================


def test_read_file_text_reads_pdf(
    tmp_path,
    mocker,
):
    file_path = (
        tmp_path
        / "document.pdf"
    )

    file_path.write_bytes(
        b"fake pdf"
    )

    page_1 = mocker.Mock()
    page_1.extract_text.return_value = (
        "Page one"
    )

    page_2 = mocker.Mock()
    page_2.extract_text.return_value = (
        "Page two"
    )

    fake_reader = mocker.Mock()

    fake_reader.pages = [
        page_1,
        page_2,
    ]

    mocker.patch(
        "rag.PdfReader",
        return_value=fake_reader,
    )

    result = rag.read_file_text(
        str(file_path)
    )

    assert result == (
        "Page one\n"
        "Page two\n"
    )


def test_read_file_text_reads_docx(
    tmp_path,
    mocker,
):
    file_path = (
        tmp_path
        / "document.docx"
    )

    file_path.write_bytes(
        b"fake docx"
    )

    mock_process = mocker.patch(
        "rag.docx2txt.process",
        return_value=(
            "Document DOCX content"
        ),
    )

    result = rag.read_file_text(
        str(file_path)
    )

    assert (
        result
        == "Document DOCX content"
    )

    mock_process.assert_called_once_with(
        str(file_path)
    )




# Test Retrieval bez numeru strony

def test_retrieve_from_rag_handles_source_without_page(
    mock_pgvector_retrieval,
):
    _, db = (
        mock_pgvector_retrieval
    )

    document = rag.RagDocument(
        id=1,
        thread_id="thread-txt",
        source_name="notes.txt",
        stored_name="stored_notes.txt",
    )

    chunk = rag.RagChunk(
        document_id=1,
        page=None,
        chunk_index=3,
        content=(
            "Content from TXT document."
        ),
        embedding=[0.1, 0.2, 0.3],
    )

    query = db.query.return_value

    query.join.return_value = query
    query.filter.return_value = query
    query.order_by.return_value = query
    query.limit.return_value = query

    query.all.return_value = [
        (
            chunk,
            document,
        )
    ]

    result = rag.retrieve_from_rag(
        query="What is in notes?",
        thread_id="thread-txt",
        k=4,
    )

    assert (
        "[Source: notes.txt]\n"
        "Content from TXT document."
        in result["context"]
    )

    assert result["sources"] == [
        {
            "source": "notes.txt",
            "page": None,
            "chunk_index": 3,
        }
    ]




# Test lazy initialization

def test_get_embeddings_is_lazily_created_and_cached(
    mocker,
):
    rag.clear_rag_resource_cache()

    mock_embeddings_class = mocker.patch(
        "rag.OpenAIEmbeddings"
    )

    mock_embeddings = mocker.Mock()

    mock_embeddings_class.return_value = (
        mock_embeddings
    )

    first = rag.get_embeddings()
    second = rag.get_embeddings()

    assert first is mock_embeddings
    assert second is mock_embeddings

    mock_embeddings_class.assert_called_once_with(
        model=rag.settings.embedding_model,
        api_key=rag.settings.openai_api_key,
    )

    rag.clear_rag_resource_cache()




# Test tylko wybrany dokument

def test_delete_document_from_rag_deletes_only_selected_document(
    tmp_path,
    monkeypatch,
    mock_pgvector_delete,
    mocker,
):
    upload_dir = (
        tmp_path
        / "uploads"
    )

    upload_dir.mkdir()

    monkeypatch.setattr(
        rag.settings,
        "upload_dir",
        upload_dir,
    )

    selected_file = (
        upload_dir
        / "uuid_selected.txt"
    )

    other_file = (
        upload_dir
        / "uuid_other.txt"
    )

    selected_file.write_text(
        "selected",
        encoding="utf-8",
    )

    other_file.write_text(
        "other",
        encoding="utf-8",
    )

    document = rag.RagDocument(
        id=10,
        thread_id="thread-1",
        source_name="selected.txt",
        stored_name="uuid_selected.txt",
    )

    db = mock_pgvector_delete

    document_query = mocker.Mock()
    chunk_query = mocker.Mock()

    db.query.side_effect = [
        document_query,
        chunk_query,
    ]

    (
        document_query
        .filter
        .return_value
        .first
        .return_value
    ) = document

    (
        chunk_query
        .filter
        .return_value
        .count
        .return_value
    ) = 2

    result = (
        rag.delete_document_from_rag(
            thread_id="thread-1",
            stored_name="uuid_selected.txt",
        )
    )

    assert selected_file.exists() is False
    assert other_file.exists() is True

    db.delete.assert_called_once_with(
        document
    )

    db.commit.assert_called_once_with()
    db.rollback.assert_not_called()
    db.close.assert_called_once_with()

    assert result == {
        "deleted_chunks": 2,
        "deleted_files": 1,
    }




def test_add_document_to_rag_rolls_back_when_database_write_fails(
    tmp_path,
    mock_pgvector_storage,
):
    file_path = (
        tmp_path
        / "rollback_test.txt"
    )

    file_path.write_text(
        "Document content for rollback test.",
        encoding="utf-8",
    )

    embeddings, db = (
        mock_pgvector_storage
    )

    db.add_all.side_effect = RuntimeError(
        "Database write failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Database write failed",
    ):
        rag.add_document_to_rag(
            file_path=str(file_path),
            thread_id="thread-rollback",
            original_filename="rollback.txt",
        )

    embeddings.embed_documents.assert_called_once()

    db.add.assert_called_once()
    db.flush.assert_called_once()
    db.add_all.assert_called_once()

    db.commit.assert_not_called()
    db.rollback.assert_called_once_with()
    db.close.assert_called_once_with()




# Test rollbacku

def test_delete_document_from_rag_rolls_back_when_database_delete_fails(
    tmp_path,
    monkeypatch,
    mock_pgvector_delete,
    mocker,
):
    upload_dir = (
        tmp_path
        / "uploads"
    )

    upload_dir.mkdir()

    monkeypatch.setattr(
        rag.settings,
        "upload_dir",
        upload_dir,
    )

    document = rag.RagDocument(
        id=1,
        thread_id="thread-fail",
        source_name="document.txt",
        stored_name="stored.txt",
    )

    db = mock_pgvector_delete

    document_query = mocker.Mock()
    chunk_query = mocker.Mock()

    db.query.side_effect = [
        document_query,
        chunk_query,
    ]

    (
        document_query
        .filter
        .return_value
        .first
        .return_value
    ) = document

    (
        chunk_query
        .filter
        .return_value
        .count
        .return_value
    ) = 1

    db.commit.side_effect = RuntimeError(
        "Database delete failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Database delete failed",
    ):
        rag.delete_document_from_rag(
            thread_id="thread-fail",
            stored_name="stored.txt",
        )

    db.rollback.assert_called_once_with()
    db.close.assert_called_once_with()