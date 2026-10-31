import pytest

from pydantic import ValidationError

from api.schemas.chat import (
    ChatStreamRequest,
)


def test_chat_request_accepts_normal_message():
    payload = ChatStreamRequest(
        message="Hello",
        thread_id="thread-123",
        model="gpt-4o-mini",
        speech_language="pl-PL",
    )

    assert payload.message == "Hello"
    assert payload.thread_id == "thread-123"


def test_chat_request_accepts_resume():
    payload = ChatStreamRequest(
        thread_id="thread-123",
        model="gpt-4o-mini",
        resume=True,
    )

    assert payload.resume is True


def test_chat_request_rejects_empty_message():
    with pytest.raises(
        ValidationError
    ):
        ChatStreamRequest(
            message="   ",
            thread_id="thread-123",
        )


def test_chat_request_rejects_missing_thread_id():
    with pytest.raises(
        ValidationError
    ):
        ChatStreamRequest(
            message="Hello",
        )


def test_chat_request_rejects_invalid_thread_id():
    with pytest.raises(
        ValidationError
    ):
        ChatStreamRequest(
            message="Hello",
            thread_id="../secret",
        )


def test_chat_request_rejects_long_message():
    with pytest.raises(
        ValidationError
    ):
        ChatStreamRequest(
            message="x" * 20_001,
            thread_id="thread-123",
        )


def test_chat_request_rejects_unknown_model():
    with pytest.raises(
        ValidationError
    ):
        ChatStreamRequest(
            message="Hello",
            thread_id="thread-123",
            model="unknown-model",
        )


def test_chat_request_rejects_unknown_field():
    with pytest.raises(
        ValidationError
    ):
        ChatStreamRequest(
            message="Hello",
            thread_id="thread-123",
            unexpected=True,
        )


def test_chat_request_rejects_message_and_resume():
    with pytest.raises(
        ValidationError
    ):
        ChatStreamRequest(
            message="Hello",
            thread_id="thread-123",
            resume=True,
        )