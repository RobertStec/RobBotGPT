import importlib
import json

import pytest
import database

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)



@pytest.fixture
def chat_router_module(
    integration_app,
):
    return importlib.import_module(
        "api.routers.chat"
    )



def parse_sse_events(response):
    """
    Convert SSE response body into a list of JSON events.
    """

    events = []

    for block in response.text.split("\n\n"):
        block = block.strip()

        if not block:
            continue

        data_lines = [
            line.removeprefix("data:").strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]

        if not data_lines:
            continue

        json_payload = "\n".join(
            data_lines
        )

        events.append(
            json.loads(json_payload)
        )

    return events



# Test zwykłego streamingu tokenów

def test_chat_stream_returns_tokens_and_done(
    api_client,
    chat_router_module,
    mocker,
):
    fake_agent = mocker.Mock()

    fake_agent.stream.return_value = [
        {
            "type": "messages",
            "data": (
                AIMessageChunk(
                    content="Hello "
                ),
                {
                    "langgraph_node": "chatbot"
                },
            ),
        },
        {
            "type": "messages",
            "data": (
                AIMessageChunk(
                    content="Robert!"
                ),
                {
                    "langgraph_node": "chatbot"
                },
            ),
        },
    ]

    mock_get_agent = mocker.patch.object(
        chat_router_module,
        "get_agent",
        return_value=fake_agent,
    )

    response = api_client.post(
        "/chat/stream",
        json={
            "message": "Hello",
            "thread_id": "thread-chat",
            "model": "gpt-4o-mini",
            "speech_language": "pl-PL",
        },
    )

    assert response.status_code == 200

    assert response.headers[
        "content-type"
    ].startswith(
        "text/event-stream"
    )

    events = parse_sse_events(
        response
    )

    assert events == [
        {
            "type": "token",
            "token": "Hello ",
        },
        {
            "type": "token",
            "token": "Robert!",
        },
        {
            "type": "done",
            "done": True,
        },
    ]

    mock_get_agent.assert_called_once_with(
        "gpt-4o-mini"
    )



# Test sprawdzenia danych przekazywanych do LangGraph

    fake_agent.stream.assert_called_once()

    stream_call = (
        fake_agent.stream.call_args
    )

    graph_input = (
        stream_call.args[0]
    )

    assert isinstance(
        graph_input["messages"][0],
        HumanMessage,
    )

    assert (
        graph_input["messages"][0].content
        == "Hello"
    )

    assert (
        graph_input["selected_model"]
        == "gpt-4o-mini"
    )

    assert (
        graph_input["speech_language"]
        == "pl-PL"
    )

    config = (
        stream_call.kwargs["config"]
    )

    assert (
        config["configurable"]["thread_id"]
        == "thread-chat"
    )

    assert (
        config["metadata"]["model"]
        == "gpt-4o-mini"
    )

    assert (
        stream_call.kwargs["version"]
        == "v2"
    )

    assert stream_call.kwargs[
        "stream_mode"
    ] == [
        "messages",
        "updates",
        "values",
        "custom",
    ]

    history = database.get_chat_history(
        "thread-chat"
    )

    assert len(history) == 2

    assert history[0].role == "user"
    assert history[0].content == "Hello"

    assert history[1].role == "assistant"
    assert history[1].content == "Hello Robert!"




# Test tool_start i tool_end

def test_chat_stream_emits_tool_start_and_tool_end(
    api_client,
    chat_router_module,
    mocker,
):
    fake_agent = mocker.Mock()

    tool_call_message = AIMessage(
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

    tool_result_message = ToolMessage(
        content="4",
        tool_call_id="call-123",
    )

    fake_agent.stream.return_value = [
        {
            "type": "updates",
            "data": {
                "chatbot": {
                    "messages": [
                        tool_call_message
                    ]
                }
            },
        },
        {
            "type": "updates",
            "data": {
                "tools": {
                    "messages": [
                        tool_result_message
                    ]
                }
            },
        },
        {
            "type": "messages",
            "data": (
                AIMessageChunk(
                    content="The result is 4."
                ),
                {
                    "langgraph_node": "chatbot"
                },
            ),
        },
    ]

    mocker.patch.object(
        chat_router_module,
        "get_agent",
        return_value=fake_agent,
    )

    response = api_client.post(
        "/chat/stream",
        json={
            "message": "Calculate 2 + 2",
            "thread_id": "thread-tool",
            "model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200

    events = parse_sse_events(
        response
    )

    assert events == [
        {
            "type": "tool_start",
            "tool": "calculator",
            "tool_call_id": "call-123",
        },
        {
            "type": "tool_end",
            "tool": "calculator",
            "tool_call_id": "call-123",
        },
        {
            "type": "token",
            "token": "The result is 4.",
        },
        {
            "type": "done",
            "done": True,
        },
    ]




# Test eventów RAG

def test_chat_stream_emits_rag_events_and_saves_sources(
    api_client,
    chat_router_module,
    mocker,
):
    fake_agent = mocker.Mock()

    sources = [
        {
            "source": "manual.pdf",
            "page": 5,
            "chunk_index": 2,
        }
    ]

    fake_agent.stream.return_value = [
        {
            "type": "custom",
            "data": {
                "type": "rag_search_start",
                "message": "Searching documents...",
            },
        },
        {
            "type": "custom",
            "data": {
                "type": "rag_search_end",
                "message": "Document Search completed",
            },
        },
        {
            "type": "custom",
            "data": {
                "type": "rag_sources",
                "sources": sources,
            },
        },
        {
            "type": "messages",
            "data": (
                AIMessageChunk(
                    content="Answer from document."
                ),
                {
                    "langgraph_node": "generate"
                },
            ),
        },
    ]

    mocker.patch.object(
        chat_router_module,
        "get_agent",
        return_value=fake_agent,
    )

    response = api_client.post(
        "/chat/stream",
        json={
            "message": (
                "What does the document say?"
            ),
            "thread_id": "thread-rag-stream",
            "model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200

    events = parse_sse_events(
        response
    )

    assert events == [
        {
            "type": "rag_search_start",
            "message": "Searching documents...",
        },
        {
            "type": "rag_search_end",
            "message": "Document Search completed",
        },
        {
            "type": "rag_sources",
            "sources": sources,
        },
        {
            "type": "token",
            "token": "Answer from document.",
        },
        {
            "type": "done",
            "done": True,
        },
    ]




# Test czy źródła naprawdę trafiły do historii

    history_response = api_client.get(
        "/history/thread-rag-stream"
    )

    assert (
        history_response.status_code
        == 200
    )

    messages = (
        history_response
        .json()["messages"]
    )

    assert len(messages) == 2

    assistant_message = (
        messages[1]
    )

    assert (
        assistant_message["content"]
        == "Answer from document."
    )

    assert (
        assistant_message["sources"]
        == sources
    )




# Test błędu podczas streamowania

def test_chat_stream_emits_error_and_done_when_agent_fails(
    api_client,
    chat_router_module,
    mocker,
):
    fake_agent = mocker.Mock()

    fake_agent.stream.side_effect = (
        RuntimeError(
            "Graph execution failed"
        )
    )

    mocker.patch.object(
        chat_router_module,
        "get_agent",
        return_value=fake_agent,
    )

    response = api_client.post(
        "/chat/stream",
        json={
            "message": "Hello",
            "thread_id": "thread-error",
            "model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200

    events = parse_sse_events(
        response
    )

    assert events == [
        {
            "type": "error",
            "error": (
                "An internal error occurred "
                "while generating the response."
            ),
            "error_code": (
                "chat_stream_failed"
            ),
        },
        {
            "type": "done",
            "done": True,
        },
    ]

    assert (
        "Graph execution failed"
        not in response.text
    )

    history = database.get_chat_history(
        "thread-error"
    )

    assert len(history) == 1

    assert history[0].role == "user"
    assert history[0].content == "Hello"




# Test niepoprawnego JSON

def test_chat_stream_rejects_invalid_json(
    api_client,
):
    response = api_client.post(
        "/chat/stream",
        content="{invalid-json",
        headers={
            "Content-Type": "application/json"
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "error": "Invalid JSON body."
    }



# Test pusta wiadomość

def test_chat_stream_rejects_empty_message(
    api_client,
):
    response = api_client.post(
        "/chat/stream",
        json={
            "message": "   ",
            "thread_id": "thread-empty",
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "error": "Message is required."
    }