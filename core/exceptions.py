import logging
from fastapi import Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


class AppError(Exception):
    """
    Base exception for errors that can be safely
    translated into an HTTP response.
    """

    status_code = 500
    error_code = "application_error"
    default_message = (
        "An unexpected application error occurred."
    )

    def __init__(
        self,
        *,
        message: str | None = None,
        internal_message: str | None = None,
    ):
        self.public_message = (
            message
            or self.default_message
        )

        self.internal_message = (
            internal_message
        )

        super().__init__(
            internal_message
            or self.public_message
        )


class UnsupportedFileTypeError(AppError):
    status_code = 400
    error_code = "unsupported_file_type"
    default_message = (
        "Unsupported file type. "
        "Upload PDF, DOCX, TXT, MD, PY, or CSV."
    )


class ConversationNotFoundError(AppError):
    status_code = 404
    error_code = "conversation_not_found"
    default_message = (
        "Conversation not found."
    )


class DocumentProcessingError(AppError):
    status_code = 500
    error_code = "document_processing_failed"
    default_message = (
        "Could not process the uploaded document."
    )


class ConversationDeletionError(AppError):
    status_code = 500
    error_code = "conversation_deletion_failed"
    default_message = (
        "Could not completely delete conversation."
    )


async def app_error_handler(
    request: Request,
    exc: AppError,
) -> JSONResponse:
    """
    Convert application exceptions into safe,
    consistent HTTP responses.
    """

    if exc.status_code >= 500:
        logger.error(
            (
                "Application error "
                "code=%s status=%d path=%s internal=%s"
            ),
            exc.error_code,
            exc.status_code,
            request.url.path,
            exc.internal_message,
        )

    else:
        logger.warning(
            (
                "Client application error "
                "code=%s status=%d path=%s"
            ),
            exc.error_code,
            exc.status_code,
            request.url.path,
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.public_message,
            "error_code": exc.error_code,
        },
    )