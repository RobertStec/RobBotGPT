import json

import pytest

from core.exceptions import (
    DocumentProcessingError,
    app_error_handler,
    RateLimitExceededError,
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