from unittest.mock import call

import pytest

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


from nodes import (
    chatbot_node,
    generate_rag_node,
    get_latest_user_message,
    is_document_question,
    retrieve_node,
    route_after_router,
    router_node,
)


# =========================================================
# is_document_question
# =========================================================

@pytest.mark.parametrize(
    "message",
    [
        "Summarize the document I uploaded.",
        "What does the uploaded PDF say?",
        "Please check the attached file.",
        "Podsumuj dokument, który wgrałem.",
        "Co znajduje się w załączniku?",
        "Sprawdź wgrany plik.",
    ],
)
def test_is_document_question_returns_true_for_document_queries(
    message,
):
    state = {
        "messages": [
            HumanMessage(content=message)
        ]
    }

    assert is_document_question(state) is True


def test_is_document_question_returns_false_for_normal_question():
    state = {
        "messages": [
            HumanMessage(
                content="Jaka jest stolica Polski?"
            )
        ]
    }

    assert is_document_question(state) is False


def test_is_document_question_returns_false_for_empty_messages():
    state = {
        "messages": []
    }

    assert is_document_question(state) is False


def test_is_document_question_ignores_ai_message():
    state = {
        "messages": [
            AIMessage(
                content="The document contains information."
            )
        ]
    }

    assert is_document_question(state) is False


# =========================================================
# router_node
# =========================================================

def test_router_node_routes_document_question_to_document():
    state = {
        "messages": [
            HumanMessage(
                content="Summarize the uploaded document."
            )
        ]
    }

    result = router_node(state)

    assert result == {
        "route": "document"
    }


def test_router_node_routes_normal_question_to_chatbot():
    state = {
        "messages": [
            HumanMessage(
                content="What is Python?"
            )
        ]
    }

    result = router_node(state)

    assert result == {
        "route": "chatbot"
    }


# =========================================================
# route_after_router
# =========================================================

@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("chatbot", "chatbot"),
        ("document", "document"),
    ],
)
def test_route_after_router_returns_selected_route(
    route,
    expected,
):
    state = {
        "messages": [],
        "route": route,
    }

    assert route_after_router(state) == expected


def test_route_after_router_defaults_to_chatbot():
    state = {
        "messages": []
    }

    assert route_after_router(state) == "chatbot"


# =========================================================
# get_latest_user_message
# =========================================================

def test_get_latest_user_message_returns_latest_human_message():
    state = {
        "messages": [
            HumanMessage(
                content="First question"
            ),
            AIMessage(
                content="First answer"
            ),
            HumanMessage(
                content="Second question"
            ),
            AIMessage(
                content="Second answer"
            ),
        ]
    }

    result = get_latest_user_message(state)

    assert result == "Second question"


def test_get_latest_user_message_returns_empty_string_when_missing():
    state = {
        "messages": [
            AIMessage(
                content="Assistant message"
            )
        ]
    }

    assert get_latest_user_message(state) == ""


# =========================================================
# retrieve_node
# =========================================================

def test_retrieve_node_returns_context_and_sources(
    mocker,
):
    state = {
        "messages": [
            HumanMessage(
                content="What does the document say?"
            )
        ]
    }

    config = {
        "configurable": {
            "thread_id": "thread-123"
        }
    }

    writer = mocker.Mock()

    mock_retrieve = mocker.patch(
        "nodes.retrieve_from_rag",
        return_value={
            "context": "Retrieved document context.",
            "sources": [
                {
                    "source": "example.pdf",
                    "page": 2,
                    "chunk_index": 4,
                }
            ],
        },
    )

    result = retrieve_node(
        state=state,
        config=config,
        writer=writer,
    )

    assert result == {
        "rag_context": "Retrieved document context.",
        "rag_sources": [
            {
                "source": "example.pdf",
                "page": 2,
                "chunk_index": 4,
            }
        ],
    }

    mock_retrieve.assert_called_once_with(
        query="What does the document say?",
        thread_id="thread-123",
        k=4,
    )

    assert writer.call_args_list == [
        call({
            "type": "rag_search_start",
            "message": "Searching documents...",
        }),
        call({
            "type": "rag_search_end",
            "message": "Document Search completed",
        }),
    ]


def test_retrieve_node_raises_error_when_thread_id_missing(
    mocker,
):
    state = {
        "messages": [
            HumanMessage(
                content="Check the document."
            )
        ]
    }

    config = {
        "configurable": {}
    }

    writer = mocker.Mock()

    mock_retrieve = mocker.patch(
        "nodes.retrieve_from_rag"
    )

    with pytest.raises(
        ValueError,
        match="thread_id is missing",
    ):
        retrieve_node(
            state=state,
            config=config,
            writer=writer,
        )

    mock_retrieve.assert_not_called()
    writer.assert_not_called()


# =========================================================
# generate_rag_node
# =========================================================

def test_generate_rag_node_uses_context_and_returns_message(
    mocker,
):
    user_message = HumanMessage(
        content="What is described in the document?"
    )

    state = {
        "messages": [
            user_message
        ],
        "rag_context": (
            "RobBotGPT is an application "
            "built with LangGraph."
        ),
        "rag_sources": [
            {
                "source": "project.pdf",
                "page": 1,
                "chunk_index": 1,
            }
        ],
    }

    expected_response = AIMessage(
        content="RobBotGPT is built with LangGraph."
    )

    llm = mocker.Mock()
    llm.invoke.return_value = expected_response

    writer = mocker.Mock()

    result = generate_rag_node(
        state=state,
        llm=llm,
        writer=writer,
    )

    assert result == {
        "messages": [
            expected_response
        ]
    }

    llm.invoke.assert_called_once()

    messages = llm.invoke.call_args.args[0]

    assert len(messages) == 2

    assert isinstance(
        messages[0],
        SystemMessage,
    )

    assert (
        "RobBotGPT is an application "
        "built with LangGraph."
        in messages[0].content
    )

    assert messages[1] == user_message

    writer.assert_called_once_with({
        "type": "rag_sources",
        "sources": [
            {
                "source": "project.pdf",
                "page": 1,
                "chunk_index": 1,
            }
        ],
    })


def test_generate_rag_node_does_not_emit_sources_when_empty(
    mocker,
):
    state = {
        "messages": [
            HumanMessage(
                content="Question about document"
            )
        ],
        "rag_context": "Some context",
        "rag_sources": [],
    }

    llm = mocker.Mock()

    llm.invoke.return_value = AIMessage(
        content="Answer"
    )

    writer = mocker.Mock()

    generate_rag_node(
        state=state,
        llm=llm,
        writer=writer,
    )

    writer.assert_not_called()


# =========================================================
# chatbot_node
# =========================================================

def test_chatbot_node_adds_system_prompt_and_calls_llm(
    mocker,
):
    user_message = HumanMessage(
        content="Hello RobBotGPT"
    )

    state = {
        "messages": [
            user_message
        ]
    }

    expected_response = AIMessage(
        content="Hello!"
    )

    llm_with_tools = mocker.Mock()

    llm_with_tools.invoke.return_value = (
        expected_response
    )

    result = chatbot_node(
        state=state,
        llm_with_tools=llm_with_tools,
        system_prompt="You are RobBotGPT.",
    )

    assert result == {
        "messages": [
            expected_response
        ]
    }

    llm_with_tools.invoke.assert_called_once()

    messages = (
        llm_with_tools
        .invoke
        .call_args
        .args[0]
    )

    assert len(messages) == 2

    assert isinstance(
        messages[0],
        SystemMessage,
    )

    assert (
        messages[0].content
        == "You are RobBotGPT."
    )

    assert messages[1] == user_message