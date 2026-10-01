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




def add_document_to_rag(file_path: str, thread_id: str):
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": Path(file_path).name
            }
        )
        for chunk in chunks
    ]

    vectorstore.add_documents(docs)

    return {
        "filename": Path(file_path).name,
        "chunks": len(docs)
    }



def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={"thread_id": thread_id}
    )

    if not docs:
        return "No relevant uploaded document content found."

    results = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        results.append(
            f"[Source {i}: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(results)



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

    # Collect physical uploaded files before deleting Chroma records
    source_files = set()

    for metadata in metadatas:
        if not metadata:
            continue

        source = metadata.get("source")

        if source:
            source_files.add(source)

    # Delete chunks / embeddings from ChromaDB
    if ids:
        vectorstore.delete(ids=ids)

    # Delete original uploaded files
    deleted_files = 0

    for source in source_files:
        file_path = Path("uploads") / source

        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            deleted_files += 1

    return {
        "deleted_chunks": len(ids),
        "deleted_files": deleted_files
    }