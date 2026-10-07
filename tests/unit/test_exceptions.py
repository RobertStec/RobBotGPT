import json

import pytest

from core.exceptions import (
    DocumentProcessingError,
    app_error_handler,
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