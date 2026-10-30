import json

import pytest

from core.exceptions import (
    DocumentProcessingError,
    app_error_handler,
    RateLimitExceededError,
    EmptyUploadError,
    UploadTooLargeError,
)


@pytest.mark.asyncio
async def test_app_error_handler_returns_safe_response(
    mocker,
):
    request = mocker.Mock()
    request.url.path = "/upload"

    mock_logger = mocker.patch(
        "core.exceptions.logger"
    )

    error = DocumentProcessingError(
        internal_message=(
            "secret database details"
        )
    )

    response = await app_error_handler(
        request,
        error,
    )

    payload = json.loads(
        response.body
    )

    mock_logger.error.assert_called_once()

    assert response.status_code == 500

    assert payload == {
        "success": False,
        "message": (
            "Could not process "
            "the uploaded document."
        ),
        "error_code": (
            "document_processing_failed"
        ),
    }

    assert (
        "secret database details"
        not in response.body.decode()
    )



def test_rate_limit_exceeded_error_sets_retry_after():
    error = RateLimitExceededError(
        retry_after=42
    )

    assert error.status_code == 429
    assert (
        error.error_code
        == "rate_limit_exceeded"
    )
    assert error.retry_after == 42
    assert error.headers == {
        "Retry-After": "42"
    }



@pytest.mark.asyncio
async def test_app_error_handler_returns_rate_limit_headers(
    mocker,
):
    request = mocker.Mock()

    request.url.path = (
        "/chat/stream"
    )

    error = RateLimitExceededError(
        retry_after=37
    )

    response = await app_error_handler(
        request,
        error,
    )

    assert response.status_code == 429

    assert (
        response.headers[
            "retry-after"
        ]
        == "37"
    )



def test_empty_upload_error():
    exc = EmptyUploadError()

    assert exc.status_code == 400
    assert exc.error_code == "empty_upload"
    assert (
        exc.public_message
        == "Uploaded file is empty."
    )


def test_upload_too_large_error():
    exc = UploadTooLargeError(
        max_bytes=10 * 1024 * 1024
    )

    assert exc.status_code == 413
    assert exc.error_code == "upload_too_large"

    assert exc.public_message == (
        "Uploaded file is too large. "
        "Maximum allowed size is 10 MB."
    )