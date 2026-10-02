from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import certifi

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader
import docx2txt

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)


# Embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    collection_name="agentic_chatbot_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)



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
        chunk_size=900,
        chunk_overlap=150
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

    vectorstore.add_documents(docs)

    return {
        "filename": source_name,
        "chunks": len(docs)
    }



def retrieve_from_rag(
    query: str,
    thread_id: str,
    k: int = 4) -> dict:

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



def delete_thread_documents(thread_id: str) -> dict:
    """
    Delete all ChromaDB documents and uploaded files
    associated with a conversation thread.
    """

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
    # Delete chunks from ChromaDB
    # -----------------------------------------

    if ids:
        vectorstore.delete(
            ids=ids
        )

    # -----------------------------------------
    # Delete uploaded physical files
    # -----------------------------------------

    deleted_files = 0

    for stored_name in stored_files:

        file_path = (
            Path("uploads")
            / stored_name
        )

        if (
            file_path.exists()
            and file_path.is_file()
        ):
            file_path.unlink()

            deleted_files += 1

    return {
        "deleted_chunks": len(ids),
        "deleted_files": deleted_files
    }