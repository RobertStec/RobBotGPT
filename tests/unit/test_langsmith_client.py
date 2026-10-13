import pytest
from pydantic import SecretStr

from core.config import settings
from evals.langsmith_client import (
    get_langsmith_client,
)


def test_get_langsmith_client_uses_settings(
    mocker,
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "langsmith_api_key",
        SecretStr("test-key"),
    )

    monkeypatch.setattr(
        settings,
        "langsmith_endpoint",
        "https://eu.api.smith.langchain.com",
    )

    monkeypatch.setattr(
        settings,
        "langsmith_workspace_id",
        None,
    )

    mock_client = mocker.patch(
        "evals.langsmith_client.Client"
    )

    get_langsmith_client()

    mock_client.assert_called_once_with(
        api_url=(
            "https://eu.api.smith.langchain.com"
        ),
        api_key="test-key",
        workspace_id=None,
    )


def test_get_langsmith_client_requires_api_key(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "langsmith_api_key",
        None,
    )

    with pytest.raises(
        RuntimeError,
        match="LANGSMITH_API_KEY",
    ):
        get_langsmith_client()