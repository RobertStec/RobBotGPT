import json
from types import SimpleNamespace

import pytest

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from services.streaming import (
    parse_message_sources,
    sse_data,
    should_stream_chunk,
    extract_text_from_chunk,
)




@pytest.mark.parametrize(
    "raw_sources",
    [
        None,
        "",
    ],
)
def test_parse_message_sources_returns_empty_list_for_empty_input(
    raw_sources,
):
    result = parse_message_sources(
        raw_sources
    )

    assert result == []


def test_parse_message_sources_parses_valid_list():
    raw_sources = json.dumps([
        {
            "source": "manual.pdf",
            "page": 3,
            "chunk_index": 7,
        }
    ])

    result = parse_message_sources(
        raw_sources
    )

    assert result == [
        {
            "source": "manual.pdf",
            "page": 3,
            "chunk_index": 7,
        }
    ]


@pytest.mark.parametrize(
    "raw_sources",
    [
        '{"source": "manual.pdf"}',
        '"manual.pdf"',
        "123",
        "true",
    ],
)
def test_parse_message_sources_returns_empty_list_for_non_list_json(
    raw_sources,
):
    result = parse_message_sources(
        raw_sources
    )

    assert result == []


def test_parse_message_sources_returns_empty_list_for_invalid_json():
    result = parse_message_sources(
        "{invalid-json"
    )

    assert result == []


# =========================================================
# sse_data
# =========================================================


def test_sse_data_formats_payload_correctly():
    payload = {
        "type": "token",
        "token": "Hello",
    }

    result = sse_data(
        payload
    )

    assert result.startswith(
        "data: "
    )

    assert result.endswith(
        "\n\n"
    )

    json_content = (
        result
        .removeprefix("data: ")
        .strip()
    )

    assert json.loads(
        json_content
    ) == payload


def test_sse_data_preserves_unicode():
    payload = {
        "type": "token",
        "token": "Wrocław — dzień dobry",
    }

    result = sse_data(
        payload
    )

    assert "Wrocław" in result
    assert "dzień dobry" in result


# =========================================================
# should_stream_chunk
# =========================================================


def test_should_stream_chunk_accepts_normal_ai_chunk():
    chunk = AIMessageChunk(
        content="Hello"
    )

    metadata = {
        "langgraph_node": "chatbot"
    }

    result = should_stream_chunk(
        chunk,
        metadata,
    )

    assert result is True


def test_should_stream_chunk_accepts_normal_ai_message():
    chunk = AIMessage(
        content="Hello"
    )

    result = should_stream_chunk(
        chunk,
        {},
    )

    assert result is True


def test_should_stream_chunk_rejects_tool_node():
    chunk = AIMessageChunk(
        content="Internal tool content"
    )

    metadata = {
        "langgraph_node": "tools"
    }

    result = should_stream_chunk(
        chunk,
        metadata,
    )

    assert result is False


def test_should_stream_chunk_rejects_tool_message():
    chunk = ToolMessage(
        content="Tool output",
        tool_call_id="call-123",
    )

    result = should_stream_chunk(
        chunk,
        {},
    )

    assert result is False


def test_should_stream_chunk_rejects_human_message():
    chunk = HumanMessage(
        content="User message"
    )

    result = should_stream_chunk(
        chunk,
        {},
    )

    assert result is False


def test_should_stream_chunk_rejects_ai_message_with_tool_calls():
    chunk = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {
                    "expression": "2 + 2"
                },
                "id": "call-123",
                "type": "tool_call",
            }
        ],
    )

    result = should_stream_chunk(
        chunk,
        {},
    )

    assert result is False


def test_should_stream_chunk_rejects_additional_tool_calls():
    chunk = AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call-123"
                }
            ]
        },
    )

    result = should_stream_chunk(
        chunk,
        {},
    )

    assert result is False


# =========================================================
# extract_text_from_chunk
# =========================================================


def test_extract_text_from_chunk_returns_string_content():
    chunk = SimpleNamespace(
        content="Hello RobBotGPT"
    )

    result = extract_text_from_chunk(
        chunk
    )

    assert result == "Hello RobBotGPT"


def test_extract_text_from_chunk_returns_empty_string_for_empty_content():
    chunk = SimpleNamespace(
        content=""
    )

    result = extract_text_from_chunk(
        chunk
    )

    assert result == ""


def test_extract_text_from_chunk_combines_list_content():
    chunk = SimpleNamespace(
        content=[
            "Hello ",
            {
                "type": "text",
                "text": "RobBotGPT"
            },
            {
                "content": "!"
            },
        ]
    )

    result = extract_text_from_chunk(
        chunk
    )

    assert result == "Hello RobBotGPT!"


def test_extract_text_from_chunk_ignores_unsupported_list_items():
    chunk = SimpleNamespace(
        content=[
            {"unknown": "value"},
            123,
            None,
            "Valid text",
        ]
    )

    result = extract_text_from_chunk(
        chunk
    )

    assert result == "Valid text"


def test_extract_text_from_chunk_returns_empty_for_unsupported_content():
    chunk = SimpleNamespace(
        content={
            "something": "unsupported"
        }
    )

    result = extract_text_from_chunk(
        chunk
    )

    assert result == ""