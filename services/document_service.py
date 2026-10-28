import logging
import uuid
from pathlib import Path

from core.config import settings
from core.exceptions import (
    AppError,
    DocumentProcessingError,
    UnsupportedFileTypeError,
    ConversationNotFoundError,
)
from database import (
    conversation_exists,
    create_or_update_conversation,
    delete_conversation,
)
from rag import (
    add_document_to_rag,
    delete_document_from_rag,
)


logger = logging.getLogger(__name__)


ALLOWED_EXTENSIONS = frozenset({
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".py",
    ".csv",
})


def _remove_file_if_exists(
    file_path: Path,
) -> None:
    """
    Best-effort filesystem cleanup.

    Cleanup errors must not hide
    the original processing error.
    """

    try:
        if (
            file_path.exists()
            and file_path.is_file()
        ):
            file_path.unlink()

    except OSError:
        logger.exception(
            "Failed to remove uploaded file stored_name=%s",
            file_path.name,
        )


def process_uploaded_document(
    *,
    filename: str,
    content: bytes,
    thread_id: str,
    user_id: int,
) -> dict:
    """
    Persist an uploaded document, ingest it into RAG,
    and create/update its user-owned conversation.

    Compensates partial failures where possible.
    """

    suffix = (
        Path(filename)
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError()

    safe_filename = (
        Path(filename)
        .name
        .replace(" ", "_")
    )

    stored_name = (
        f"{uuid.uuid4()}_"
        f"{safe_filename}"
    )

    file_path = (
        settings.upload_dir
        / stored_name
    )

    conversation_preexisting = (
        conversation_exists(
            thread_id,
            user_id=user_id,
        )
    )

    ownership_established = False
    rag_attempted = False

    try:
        conversation_ready = (
            create_or_update_conversation(
                thread_id,
                "Uploaded document",
                user_id=user_id,
            )
        )

        if not conversation_ready:
            raise ConversationNotFoundError()

        ownership_established = True

        file_path.write_bytes(
            content
        )

        rag_attempted = True

        result = add_document_to_rag(
            file_path=str(file_path),
            thread_id=thread_id,
            original_filename=filename,
            user_id=user_id,
        )

        return {
            "filename": filename,
            "chunks": result["chunks"],
        }

    except Exception as exc:

        if rag_attempted:
            try:
                delete_document_from_rag(
                    thread_id=thread_id,
                    stored_name=stored_name,
                    user_id=user_id,
                )

            except Exception:
                logger.exception(
                    (
                        "Document rollback failed "
                        "user_id=%s thread_id=%s "
                        "stored_name=%s"
                    ),
                    user_id,
                    thread_id,
                    stored_name,
                )

        _remove_file_if_exists(
            file_path
        )

        if (
            ownership_established
            and not conversation_preexisting
        ):
            try:
                delete_conversation(
                    thread_id,
                    user_id=user_id,
                )

            except Exception:
                logger.exception(
                    (
                        "Conversation rollback failed "
                        "user_id=%s thread_id=%s"
                    ),
                    user_id,
                    thread_id,
                )

        if isinstance(
            exc,
            AppError,
        ):
            raise

        raise DocumentProcessingError(
            internal_message=str(exc)
        ) from exc