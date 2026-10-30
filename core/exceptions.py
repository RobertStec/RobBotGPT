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
        headers: dict[str, str] | None = None,
    ):
        self.public_message = (
            message
            or self.default_message
        )

        self.internal_message = (
            internal_message
        )

        self.headers = (
            headers
            or {}
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


class InvalidUploadFilenameError(AppError):
    status_code = 400
    error_code = "invalid_upload_filename"
    default_message = (
        "Uploaded file has an invalid filename."
    )


class InvalidUploadContentError(AppError):
    status_code = 400
    error_code = "invalid_upload_content"
    default_message = (
        "Uploaded file content does not match "
        "the expected file type."
    )


class EmptyUploadError(AppError):
    status_code = 400
    error_code = "empty_upload"
    default_message = (
        "Uploaded file is empty."
    )


class UploadTooLargeError(AppError):
    status_code = 413
    error_code = "upload_too_large"
    default_message = (
        "Uploaded file is too large."
    )

    def __init__(
        self,
        *,
        max_bytes: int,
    ):
        max_megabytes = (
            max_bytes
            / (1024 * 1024)
        )

        super().__init__(
            message=(
                "Uploaded file is too large. "
                "Maximum allowed size is "
                f"{max_megabytes:g} MB."
            )
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


class RateLimitExceededError(AppError):
    status_code = 429
    error_code = "rate_limit_exceeded"
    default_message = (
        "Too many requests. "
        "Please try again later."
    )

    def __init__(
        self,
        *,
        retry_after: int,
    ):
        self.retry_after = max(
            int(retry_after),
            1,
        )

        super().__init__(
            headers={
                "Retry-After": str(
                    self.retry_after
                )
            }
        )


class UserAlreadyExistsError(AppError):
    status_code = 409
    error_code = "user_already_exists"
    default_message = (
        "A user with this email already exists."
    )


class InvalidCredentialsError(AppError):
    status_code = 401
    error_code = "invalid_credentials"
    default_message = (
        "Invalid email or password."
    )


class AuthenticationRequiredError(AppError):
    status_code = 401
    error_code = "authentication_required"
    default_message = (
        "Authentication required."
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
        headers=exc.headers,
    )