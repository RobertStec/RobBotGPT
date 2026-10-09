from core.config import settings

from pathlib import Path
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader
import docx2txt

from database import (
    RagChunk,
    RagDocument,
    create_session,
)




settings.upload_dir.mkdir(
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




def clear_rag_resource_cache() -> None:
    """
    Clear cached RAG infrastructure.
    """

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

    texts = [
        doc.page_content
        for doc in docs
    ]

    embeddings = (
        get_embeddings()
        .embed_documents(texts)
    )

    if len(embeddings) != len(docs):
        raise RuntimeError(
            "Embedding count does not match "
            "document chunk count."
        )


    db = create_session()

    try:
        rag_document = RagDocument(
            thread_id=thread_id,
            source_name=source_name,
            stored_name=stored_name,
        )

        db.add(rag_document)

        db.flush()

        rag_chunks = [
            RagChunk(
                document_id=rag_document.id,
                page=doc.metadata.get("page"),
                chunk_index=index,
                content=doc.page_content,
                embedding=embedding,
            )
            for index, (
                doc,
                embedding,
            ) in enumerate(
                zip(
                    docs,
                    embeddings,
                    strict=True,
                ),
                start=1,
            )
        ]

        db.add_all(rag_chunks)

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

    return {
        "filename": source_name,
        "chunks": len(docs),
    }





def retrieve_from_rag(
    query: str,
    thread_id: str,
    k: int = 4,
) -> dict:

    query_embedding = (
        get_embeddings()
        .embed_query(query)
    )

    db = create_session()

    try:
        rows = (
            db.query(
                RagChunk,
                RagDocument,
            )
            .join(
                RagDocument,
                RagChunk.document_id
                == RagDocument.id,
            )
            .filter(
                RagDocument.thread_id
                == thread_id
            )
            .order_by(
                RagChunk.embedding.cosine_distance(
                    query_embedding
                )
            )
            .limit(k)
            .all()
        )

        if not rows:
            return {
                "context": "",
                "sources": [],
            }

        context_parts = []
        sources = []

        for chunk, document in rows:

            source = (
                document.source_name
                or "uploaded document"
            )

            page = chunk.page
            chunk_index = (
                chunk.chunk_index
            )

            if page is not None:
                context_parts.append(
                    f"[Source: {source}, "
                    f"page: {page}]\n"
                    f"{chunk.content}"
                )
            else:
                context_parts.append(
                    f"[Source: {source}]\n"
                    f"{chunk.content}"
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
                else source["chunk_index"],
            )

            if key in seen:
                continue

            seen.add(key)
            unique_sources.append(
                source
            )

        return {
            "context": "\n\n".join(
                context_parts
            ),
            "sources": unique_sources,
        }

    finally:
        db.close()




def delete_document_from_rag(
    thread_id: str,
    stored_name: str,
) -> dict:
    """
    Delete one uploaded document, its RAG chunks,
    and its physical uploaded file.

    RagChunk rows are removed by PostgreSQL through
    ON DELETE CASCADE.
    """

    db = create_session()

    try:
        document = (
            db.query(RagDocument)
            .filter(
                RagDocument.thread_id
                == thread_id,
                RagDocument.stored_name
                == stored_name,
            )
            .first()
        )

        deleted_chunks = 0

        if document is not None:
            deleted_chunks = (
                db.query(RagChunk)
                .filter(
                    RagChunk.document_id
                    == document.id
                )
                .count()
            )

        # Delete physical file first.
        # If this fails, database data remains
        # available for a safe retry.
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

        if document is not None:
            db.delete(document)

            # RagChunk rows are deleted through:
            # ON DELETE CASCADE
            db.commit()

        return {
            "deleted_chunks": deleted_chunks,
            "deleted_files": deleted_files,
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()




def delete_thread_documents(
    thread_id: str,
) -> dict:
    """
    Delete all uploaded documents, RAG chunks,
    and physical files associated with a thread.
    """

    db = create_session()

    try:
        documents = (
            db.query(RagDocument)
            .filter(
                RagDocument.thread_id
                == thread_id
            )
            .all()
        )

        if not documents:
            return {
                "deleted_chunks": 0,
                "deleted_files": 0,
            }

        document_ids = [
            document.id
            for document in documents
        ]

        deleted_chunks = (
            db.query(RagChunk)
            .filter(
                RagChunk.document_id.in_(
                    document_ids
                )
            )
            .count()
        )

        stored_names = {
            document.stored_name
            for document in documents
            if document.stored_name
        }

        # Delete files first.
        # Database rows remain intact if filesystem
        # cleanup fails, which makes retry possible.
        deleted_files = 0

        for stored_name in stored_names:
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

        for document in documents:
            db.delete(document)

        db.commit()

        return {
            "deleted_chunks": deleted_chunks,
            "deleted_files": deleted_files,
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()