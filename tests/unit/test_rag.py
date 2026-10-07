import pytest

from langchain_core.documents import Document

import rag



@pytest.fixture
def mock_vectorstore(mocker):
    store = mocker.Mock()

    mocker.patch.object(
        rag,
        "get_vectorstore",
        return_value=store,
    )

    return store




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
    mock_vectorstore,
):
    file_path = tmp_path / "stored_document.txt"

    file_path.write_text(
        "RobBotGPT uses LangGraph and LangChain.",
        encoding="utf-8",
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

    mock_vectorstore.add_documents.assert_called_once()

    documents = (
        mock_vectorstore
        .add_documents
        .call_args
        .args[0]
    )

    assert len(documents) == 1

    document = documents[0]

    assert (
        document.page_content
        == "RobBotGPT uses LangGraph and LangChain."
    )

    assert document.metadata["thread_id"] == "thread-123"
    assert document.metadata["source"] == "project.txt"

    assert (
        document.metadata["stored_name"]
        == "stored_document.txt"
    )

    assert document.metadata["chunk_index"] == 1  




# Test chunkingu

def test_add_document_to_rag_splits_long_document(
    tmp_path,
    mock_vectorstore,
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

    documents = (
        mock_vectorstore
        .add_documents
        .call_args
        .args[0]
    )

    assert result["chunks"] > 1
    assert len(documents) > 1

    assert [
        doc.metadata["chunk_index"]
        for doc in documents
    ] == list(
        range(1, len(documents) + 1)
    )



# Pusty dokument

def test_add_document_to_rag_rejects_empty_document(
    tmp_path,
    mock_vectorstore,
):
    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "   \n   ",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="No text could be extracted",
    ):
        rag.add_document_to_rag(
            file_path=str(file_path),
            thread_id="thread-empty",
        )

    mock_vectorstore.add_documents.assert_not_called()




# Test zachowanie numerów stron PDF

def test_add_pdf_preserves_page_numbers(
    tmp_path,
    mock_vectorstore,
    mocker,
):
    file_path = tmp_path / "document.pdf"

    file_path.write_bytes(b"fake pdf")

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

    documents = (
        mock_vectorstore
        .add_documents
        .call_args
        .args[0]
    )

    assert result["chunks"] == 2

    assert len(documents) == 2

    assert documents[0].metadata["page"] == 1
    assert documents[1].metadata["page"] == 2

    assert (
        documents[0].metadata["thread_id"]
        == "thread-pdf"
    )

    assert (
        documents[1].metadata["source"]
        == "manual.pdf"
    )




# Test Retrieval - poprawny filtr thread_id

def test_retrieve_from_rag_filters_by_thread_id(
    mock_vectorstore,
):
    mock_vectorstore.similarity_search.return_value = [
        Document(
            page_content="LangGraph manages agent workflows.",
            metadata={
                "thread_id": "thread-123",
                "source": "project.pdf",
                "page": 2,
                "chunk_index": 3,
            },
        )
    ]

    result = rag.retrieve_from_rag(
        query="What does LangGraph do?",
        thread_id="thread-123",
        k=4,
    )

    mock_vectorstore.similarity_search.assert_called_once_with(
        "What does LangGraph do?",
        k=4,
        filter={
            "thread_id": "thread-123"
        },
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



# Test Retrieval bez wyników

def test_retrieve_from_rag_returns_empty_result_when_no_documents(
    mock_vectorstore,
):
    mock_vectorstore.similarity_search.return_value = []

    result = rag.retrieve_from_rag(
        query="Unknown information",
        thread_id="thread-empty",
        k=4,
    )

    assert result == {
        "context": "",
        "sources": [],
    }



# Test deduplikacji źródeł

def test_retrieve_from_rag_deduplicates_sources(
    mock_vectorstore,
):
    mock_vectorstore.similarity_search.return_value = [
        Document(
            page_content="First chunk",
            metadata={
                "source": "manual.pdf",
                "page": 5,
                "chunk_index": 1,
            },
        ),
        Document(
            page_content="Second chunk",
            metadata={
                "source": "manual.pdf",
                "page": 5,
                "chunk_index": 2,
            },
        ),
    ]

    result = rag.retrieve_from_rag(
        query="question",
        thread_id="thread-123",
    )

    assert len(result["sources"]) == 1

    assert result["sources"][0]["source"] == "manual.pdf"
    assert result["sources"][0]["page"] == 5

    assert "First chunk" in result["context"]
    assert "Second chunk" in result["context"]




# Test Usuwania dokumentów dla konkretnego thread_id

def test_delete_thread_documents_removes_chunks_and_files(
    tmp_path,
    monkeypatch,
    mock_vectorstore,
):
    monkeypatch.chdir(tmp_path)

    uploads = tmp_path / "uploads"
    uploads.mkdir()

    file_1 = uploads / "stored_1.pdf"
    file_2 = uploads / "stored_2.txt"

    file_1.write_bytes(b"pdf")
    file_2.write_text(
        "text",
        encoding="utf-8",
    )

    mock_vectorstore.get.return_value = {
        "ids": [
            "chunk-1",
            "chunk-2",
        ],
        "metadatas": [
            {
                "stored_name": "stored_1.pdf",
                "source": "original.pdf",
            },
            {
                "stored_name": "stored_2.txt",
                "source": "original.txt",
            },
        ],
    }

    result = rag.delete_thread_documents(
        "thread-delete"
    )

    mock_vectorstore.get.assert_called_once_with(
        where={
            "thread_id": "thread-delete"
        },
        include=["metadatas"],
    )

    mock_vectorstore.delete.assert_called_once_with(
        ids=[
            "chunk-1",
            "chunk-2",
        ]
    )

    assert file_1.exists() is False
    assert file_2.exists() is False

    assert result == {
        "deleted_chunks": 2,
        "deleted_files": 2,
    }



# Brak dokumentów do usunięcia

def test_delete_thread_documents_handles_empty_thread(
    tmp_path,
    monkeypatch,
    mock_vectorstore,
):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "uploads").mkdir()

    mock_vectorstore.get.return_value = {
        "ids": [],
        "metadatas": [],
    }

    result = rag.delete_thread_documents(
        "unknown-thread"
    )

    mock_vectorstore.delete.assert_not_called()

    assert result == {
        "deleted_chunks": 0,
        "deleted_files": 0,
    }



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
    mock_vectorstore,
):
    mock_vectorstore.similarity_search.return_value = [
        Document(
            page_content=(
                "Content from TXT document."
            ),
            metadata={
                "source": "notes.txt",
                "chunk_index": 3,
            },
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




# Test Backward compatibility podczas usuwania

def test_delete_thread_documents_supports_old_source_metadata(
    tmp_path,
    monkeypatch,
    mock_vectorstore,
):
    monkeypatch.chdir(
        tmp_path
    )

    uploads = (
        tmp_path
        / "uploads"
    )

    uploads.mkdir()

    old_file = (
        uploads
        / "legacy_document.txt"
    )

    old_file.write_text(
        "legacy content",
        encoding="utf-8",
    )

    mock_vectorstore.get.return_value = {
        "ids": [
            "chunk-old"
        ],
        "metadatas": [
            {
                # Old documents do not contain
                # stored_name.
                "source": (
                    "legacy_document.txt"
                )
            }
        ],
    }

    result = rag.delete_thread_documents(
        "thread-old"
    )

    mock_vectorstore.delete.assert_called_once_with(
        ids=[
            "chunk-old"
        ]
    )

    assert old_file.exists() is False

    assert result == {
        "deleted_chunks": 1,
        "deleted_files": 1,
    }



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



# Test cache vectorstore

def test_get_vectorstore_is_lazily_created_and_cached(
    tmp_path,
    monkeypatch,
    mocker,
):
    rag.clear_rag_resource_cache()

    monkeypatch.setattr(
        rag.settings,
        "chroma_dir",
        tmp_path / "chroma",
    )

    mock_embeddings = mocker.Mock()

    mocker.patch(
        "rag.get_embeddings",
        return_value=mock_embeddings,
    )

    mock_chroma_class = mocker.patch(
        "rag.Chroma"
    )

    mock_store = mocker.Mock()

    mock_chroma_class.return_value = (
        mock_store
    )

    first = rag.get_vectorstore()
    second = rag.get_vectorstore()

    assert first is mock_store
    assert second is mock_store

    mock_chroma_class.assert_called_once_with(
        collection_name="agentic_chatbot_docs",
        embedding_function=mock_embeddings,
        persist_directory=str(
            tmp_path / "chroma"
        ),
    )

    rag.clear_rag_resource_cache()




# Test tylko wybrany dokument

def test_delete_document_from_rag_deletes_only_selected_document(
    tmp_path,
    monkeypatch,
    mock_vectorstore,
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

    mock_vectorstore.get.return_value = {
        "ids": [
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        "metadatas": [
            {
                "thread_id": "thread-1",
                "stored_name": (
                    "uuid_selected.txt"
                ),
            },
            {
                "thread_id": "thread-1",
                "stored_name": (
                    "uuid_selected.txt"
                ),
            },
            {
                "thread_id": "thread-1",
                "stored_name": (
                    "uuid_other.txt"
                ),
            },
        ],
    }

    result = (
        rag.delete_document_from_rag(
            thread_id="thread-1",
            stored_name=(
                "uuid_selected.txt"
            ),
        )
    )

    mock_vectorstore.get.assert_called_once_with(
        where={
            "thread_id": "thread-1"
        },
        include=[
            "metadatas"
        ],
    )

    mock_vectorstore.delete.assert_called_once_with(
        ids=[
            "chunk-1",
            "chunk-2",
        ]
    )

    assert (
        selected_file.exists()
        is False
    )

    assert (
        other_file.exists()
        is True
    )

    assert result == {
        "deleted_chunks": 2,
        "deleted_files": 1,
    }