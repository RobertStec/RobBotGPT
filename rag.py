from core.config import settings

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from functools import lru_cache

from pypdf import PdfReader
import docx2txt





settings.upload_dir.mkdir(
    exist_ok=True
)

settings.chroma_dir.mkdir(
    exist_ok=True
)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """
    Create and cache the embedding model.

    The object is initialized lazily on first use,
    not during module import.
    """

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Create and cache the Chroma vector store.

    Initialization happens only when RAG
    functionality is actually used.
    """

    settings.chroma_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma(
        collection_name="agentic_chatbot_docs",
        embedding_function=get_embeddings(),
        persist_directory=str(
            settings.chroma_dir
        ),
    )


def clear_rag_resource_cache() -> None:
    """
    Clear cached RAG infrastructure.

    Mainly useful for tests and controlled
    reinitialization.
    """

    get_vectorstore.cache_clear()
    get_embeddings.cache_clear()



def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")




def add_document_to_rag(
    file_path: str,
    thread_id: str,
    original_filename: str | None = None,
):
    path = Path(file_path)

    source_name = (
        original_filename
        or path.name
    )

    stored_name = path.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )

    source_documents = []

    # =========================================
    # PDF - preserve page numbers
    # =========================================

    if path.suffix.lower() == ".pdf":

        reader = PdfReader(file_path)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            if not text.strip():
                continue

            source_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "thread_id": thread_id,
                        "source": source_name,
                        "stored_name": stored_name,
                        "page": page_number,
                    }
                )
            )

    # =========================================
    # Other file types
    # =========================================

    else:

        text = read_file_text(file_path)

        if not text.strip():
            raise ValueError(
                "No text could be extracted from this file."
            )

        source_documents.append(
            Document(
                page_content=text,
                metadata={
                    "thread_id": thread_id,
                    "source": source_name,
                    "stored_name": stored_name,
                }
            )
        )

    if not source_documents:
        raise ValueError(
            "No text could be extracted from this file."
        )

    # split_documents preserves metadata
    docs = splitter.split_documents(
        source_documents
    )

    for index, doc in enumerate(
        docs,
        start=1
    ):
        doc.metadata["chunk_index"] = index

    vectorstore = get_vectorstore()

    vectorstore.add_documents(docs)

    return {
        "filename": source_name,
        "chunks": len(docs)
    }



def retrieve_from_rag(
    query: str,
    thread_id: str,
    k: int = 4) -> dict:

    vectorstore = get_vectorstore()

    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={
            "thread_id": thread_id
        }
    )

    if not docs:
        return {
            "context": "",
            "sources": []
        }

    context_parts = []
    sources = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "uploaded document"
        )

        page = doc.metadata.get("page")

        chunk_index = doc.metadata.get(
            "chunk_index"
        )

        if page is not None:
            context_parts.append(
                f"[Source: {source}, page: {page}]\n"
                f"{doc.page_content}"
            )
        else:
            context_parts.append(
                f"[Source: {source}]\n"
                f"{doc.page_content}"
            )

        sources.append({
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
        })

    unique_sources = []

    seen = set()

    for source in sources:

        key = (
            source["source"],
            source["page"]
                if source["page"] is not None
                else source["chunk_index"]
        )

        if key in seen:
            continue

        seen.add(key)
        unique_sources.append(source)

    return {
        "context": "\n\n".join(context_parts),
        "sources": unique_sources
    }




def delete_document_from_rag(
    thread_id: str,
    stored_name: str,
) -> dict:
    """
    Delete one uploaded document and its chunks
    without affecting other documents in the thread.
    """

    vectorstore = get_vectorstore()

    result = vectorstore.get(
        where={
            "thread_id": thread_id
        },
        include=[
            "metadatas"
        ],
    )

    ids = (
        result.get("ids", [])
        or []
    )

    metadatas = (
        result.get("metadatas", [])
        or []
    )

    ids_to_delete = []

    for document_id, metadata in zip(
        ids,
        metadatas,
    ):
        if not metadata:
            continue

        metadata_stored_name = (
            metadata.get("stored_name")
        )

        if not metadata_stored_name:
            metadata_stored_name = (
                metadata.get("source")
            )

        if (
            metadata_stored_name
            == stored_name
        ):
            ids_to_delete.append(
                document_id
            )

    if ids_to_delete:
        vectorstore.delete(
            ids=ids_to_delete
        )

    file_path = (
        settings.upload_dir
        / stored_name
    )

    deleted_files = 0

    if (
        file_path.exists()
        and file_path.is_file()
    ):
        file_path.unlink()

        deleted_files = 1

    return {
        "deleted_chunks": len(
            ids_to_delete
        ),
        "deleted_files": deleted_files,
    }





def delete_thread_documents(thread_id: str) -> dict:
    """
    Delete all ChromaDB documents and uploaded files
    associated with a conversation thread.
    """

    vectorstore = get_vectorstore()

    result = vectorstore.get(
        where={"thread_id": thread_id},
        include=["metadatas"]
    )

    ids = result.get("ids", []) or []
    metadatas = result.get("metadatas", []) or []

    # -----------------------------------------
    # Collect physical uploaded file names
    # -----------------------------------------

    stored_files = set()

    for metadata in metadatas:

        if not metadata:
            continue

        # New metadata format
        stored_name = metadata.get(
            "stored_name"
        )

        # Backward compatibility with documents
        # uploaded before stored_name was introduced.
        if not stored_name:
            stored_name = metadata.get(
                "source"
            )

        if stored_name:
            stored_files.add(
                stored_name
            )


    # -----------------------------------------
    # Delete uploaded physical files
    # -----------------------------------------

    deleted_files = 0

    for stored_name in stored_files:

        file_path = (
            settings.upload_dir
            / stored_name
        )

        if (
            file_path.exists()
            and file_path.is_file()
        ):
            file_path.unlink()

            deleted_files += 1


    # -----------------------------------------
    # Delete chunks from ChromaDB
    # -----------------------------------------

    if ids:
        vectorstore.delete(
            ids=ids
        )       
        

    return {
        "deleted_chunks": len(ids),
        "deleted_files": deleted_files
    }