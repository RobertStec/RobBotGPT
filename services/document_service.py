import logging
import uuid
from pathlib import Path

from core.config import settings
from core.exceptions import (
    AppError,
    DocumentProcessingError,
    UnsupportedFileTypeError,
)
from database import (
    create_or_update_conversation,
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
) -> dict:
    """
    Persist an uploaded document, ingest it into RAG,
    and create/update its conversation.

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

    rag_attempted = False

    try:
        file_path.write_bytes(
            content
        )

        rag_attempted = True

        result = add_document_to_rag(
            file_path=str(file_path),
            thread_id=thread_id,
            original_filename=filename,
        )

        create_or_update_conversation(
            thread_id,
            "Uploaded document",
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
                )

            except Exception:
                logger.exception(
                    (
                        "Document rollback failed "
                        "thread_id=%s stored_name=%s"
                    ),
                    thread_id,
                    stored_name,
                )

                _remove_file_if_exists(
                    file_path
                )

        else:
            _remove_file_if_exists(
                file_path
            )

        if isinstance(
            exc,
            AppError,
        ):
            raise

        raise DocumentProcessingError(
            internal_message=str(exc)
        ) from exc