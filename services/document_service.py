import io
import zipfile
import logging
import uuid
import stat

from pypdf import PdfReader

from pathlib import (
    Path,
    PurePosixPath,
)

from core.config import settings
from core.exceptions import (
    AppError,
    DocumentProcessingError,
    UnsupportedFileTypeError,
    ConversationNotFoundError,
    InvalidUploadFilenameError,
    InvalidUploadContentError,
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


TEXT_EXTENSIONS = frozenset({
    ".txt",
    ".md",
    ".py",
    ".csv",
})


DOCX_REQUIRED_MEMBERS = frozenset({
    "[Content_Types].xml",
    "_rels/.rels",
    "word/document.xml",
})

MAX_DOCX_MEMBERS = 1000

MAX_DOCX_UNCOMPRESSED_BYTES = (
    50 * 1024 * 1024
)

MAX_DOCX_COMPRESSION_RATIO = 200.0



def _validate_docx_archive(
    archive: zipfile.ZipFile,
) -> None:
    entries = archive.infolist()

    if (
        not entries
        or len(entries) > MAX_DOCX_MEMBERS
    ):
        raise InvalidUploadContentError()

    total_uncompressed = 0
    members = set()

    for entry in entries:
        normalized_name = (
            entry.filename
            .replace("\\", "/")
        )

        path = PurePosixPath(
            normalized_name
        )

        if (
            not normalized_name
            or "\x00" in normalized_name
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise InvalidUploadContentError()

        if (
            path.parts
            and ":" in path.parts[0]
        ):
            raise InvalidUploadContentError()

        if normalized_name in members:
            raise InvalidUploadContentError()

        members.add(
            normalized_name
        )

        # Reject encrypted ZIP entries.
        if entry.flag_bits & 0x1:
            raise InvalidUploadContentError()

        unix_mode = (
            entry.external_attr >> 16
        )

        if stat.S_ISLNK(unix_mode):
            raise InvalidUploadContentError()

        if entry.is_dir():
            continue

        total_uncompressed += (
            entry.file_size
        )

        if (
            total_uncompressed
            > MAX_DOCX_UNCOMPRESSED_BYTES
        ):
            raise InvalidUploadContentError()

        if entry.file_size > 0:
            if entry.compress_size <= 0:
                raise InvalidUploadContentError()

            compression_ratio = (
                entry.file_size
                / entry.compress_size
            )

            if (
                compression_ratio
                > MAX_DOCX_COMPRESSION_RATIO
            ):
                raise InvalidUploadContentError()

    if not DOCX_REQUIRED_MEMBERS.issubset(
        members
    ):
        raise InvalidUploadContentError()




def validate_upload_content(
    filename: str,
    content: bytes,
) -> None:
    suffix = (
        Path(filename)
        .suffix
        .lower()
    )

    if suffix == ".pdf":
        try:
            reader = PdfReader(
                io.BytesIO(content)
            )

            # Force basic PDF structure parsing.
            len(reader.pages)

        except Exception as exc:
            raise InvalidUploadContentError() from exc

        return

    if suffix == ".docx":
        try:
            with zipfile.ZipFile(
                io.BytesIO(content)
            ) as archive:
                _validate_docx_archive(
                    archive
                )

        except InvalidUploadContentError:
            raise

        except (
            zipfile.BadZipFile,
            OSError,
        ) as exc:
            raise InvalidUploadContentError() from exc

        return

    if suffix in TEXT_EXTENSIONS:
        if b"\x00" in content:
            raise InvalidUploadContentError()

        try:
            text = content.decode(
                "utf-8-sig"
            )

        except UnicodeDecodeError as exc:
            raise InvalidUploadContentError() from exc

        if not text.strip():
            raise InvalidUploadContentError()

        return

    raise UnsupportedFileTypeError()


MAX_UPLOAD_FILENAME_LENGTH = 180


def validate_upload_filename(
    filename: str,
) -> str:
    filename = filename.strip()

    if not filename:
        raise InvalidUploadFilenameError()

    if len(filename) > MAX_UPLOAD_FILENAME_LENGTH:
        raise InvalidUploadFilenameError()

    if (
        "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise InvalidUploadFilenameError()

    if any(
        ord(character) < 32
        for character in filename
    ):
        raise InvalidUploadFilenameError()

    suffix = (
        Path(filename)
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError()

    return filename.replace(" ", "_")




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

    safe_filename = (
        validate_upload_filename(
            filename
        )
    )

    validate_upload_content(
        filename,
        content,
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