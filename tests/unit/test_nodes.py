from core.config import settings
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
    rerank_node,
    retrieve_node,
    rewrite_query_node,
    route_after_router,
    router_node,
    no_answer_node,
    route_after_rerank,
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
# rewrite_query_node
# =========================================================

def test_rewrite_query_node_rewrites_question_with_context(
    mocker,
):
    llm = mocker.Mock()

    llm.invoke.return_value = AIMessage(
        content=(
            "Jaki termin realizacji "
            "projektu Atlas podano w dokumencie?"
        )
    )

    state = {
        "messages": [
            HumanMessage(
                content=(
                    "W dokumencie opisano "
                    "projekt Atlas."
                )
            ),
            AIMessage(
                content=(
                    "Tak, dokument zawiera "
                    "informacje o projekcie Atlas."
                )
            ),
            HumanMessage(
                content=(
                    "A jaki był jego termin?"
                )
            ),
        ]
    }

    result = rewrite_query_node(
        state=state,
        llm=llm,
    )

    assert result == {
        "rag_query": (
            "Jaki termin realizacji "
            "projektu Atlas podano w dokumencie?"
        )
    }

    llm.invoke.assert_called_once()




def test_rewrite_query_node_falls_back_to_original_query(
    mocker,
):
    llm = mocker.Mock()

    llm.invoke.return_value = AIMessage(
        content="   "
    )

    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Jaki jest termin realizacji "
                    "w dokumencie?"
                )
            )
        ]
    }

    result = rewrite_query_node(
        state=state,
        llm=llm,
    )

    assert result == {
        "rag_query": (
            "Jaki jest termin realizacji "
            "w dokumencie?"
        )
    }




def test_rewrite_query_node_handles_missing_user_query(
    mocker,
):
    llm = mocker.Mock()

    state = {
        "messages": [
            AIMessage(
                content="Assistant message"
            )
        ]
    }

    result = rewrite_query_node(
        state=state,
        llm=llm,
    )

    assert result == {
        "rag_query": ""
    }

    llm.invoke.assert_not_called()


# =========================================================
# retrieve_node
# =========================================================

def test_retrieve_node_returns_candidates(
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
            "thread_id": "thread-123",
            "user_id": 101,
        }
    }

    writer = mocker.Mock()

    candidates = [
        {
            "chunk_id": 1,
            "document_id": 10,
            "content": "First chunk",
            "source": "example.pdf",
            "stored_name": "example.pdf",
            "page": 2,
            "chunk_index": 1,
            "distance": 0.12,
        },
        {
            "chunk_id": 2,
            "document_id": 10,
            "content": "Second chunk",
            "source": "example.pdf",
            "stored_name": "example.pdf",
            "page": 3,
            "chunk_index": 2,
            "distance": 0.20,
        },
    ]

    mock_retrieve = mocker.patch(
        "nodes.retrieve_candidates_from_rag",
        return_value=candidates,
    )

    result = retrieve_node(
        state=state,
        config=config,
        writer=writer,
    )

    assert result == {
        "rag_candidates": candidates
    }

    mock_retrieve.assert_called_once_with(
        query="What does the document say?",
        thread_id="thread-123",
        user_id=101,
        k=settings.rag_retrieval_top_k,
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




def test_retrieve_node_uses_rewritten_query(
    mocker,
):
    candidates = [
        {
            "chunk_id": 1,
            "document_id": 10,
            "content": "Deadline chunk",
            "source": "project.pdf",
            "stored_name": "project.pdf",
            "page": 4,
            "chunk_index": 1,
            "distance": 0.11,
        }
    ]

    mock_retrieve = mocker.patch(
        "nodes.retrieve_candidates_from_rag",
        return_value=candidates,
    )

    writer = mocker.Mock()

    state = {
        "messages": [
            HumanMessage(
                content="A jaki był jego termin?"
            )
        ],
        "rag_query": (
            "Jaki termin realizacji projektu "
            "Atlas podano w dokumencie?"
        ),
    }

    config = {
        "configurable": {
            "thread_id": "thread-123",
            "user_id": 101,
        }
    }

    result = retrieve_node(
        state=state,
        config=config,
        writer=writer,
    )

    mock_retrieve.assert_called_once_with(
        query=(
            "Jaki termin realizacji projektu "
            "Atlas podano w dokumencie?"
        ),
        thread_id="thread-123",
        user_id=101,
        k=settings.rag_retrieval_top_k,
    )

    assert result == {
        "rag_candidates": candidates
    }




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
        "nodes.retrieve_candidates_from_rag"
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




def test_retrieve_node_raises_error_when_user_id_missing(
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
        "configurable": {
            "thread_id": "thread-123"
        }
    }

    writer = mocker.Mock()

    mock_retrieve = mocker.patch(
        "nodes.retrieve_candidates_from_rag"
    )

    with pytest.raises(
        ValueError,
        match="user_id is missing",
    ):
        retrieve_node(
            state=state,
            config=config,
            writer=writer,
        )

    mock_retrieve.assert_not_called()
    writer.assert_not_called()



def test_retrieve_node_uses_conversation_thread_id(
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
        "configurable": {
            "thread_id": (
                "user:42:thread:thread-123"
            ),
            "conversation_thread_id": (
                "thread-123"
            ),
            "user_id": 42,
        }
    }

    writer = mocker.Mock()

    mock_retrieve = mocker.patch(
        "nodes.retrieve_candidates_from_rag",
        return_value=[],
    )

    retrieve_node(
        state=state,
        config=config,
        writer=writer,
    )

    mock_retrieve.assert_called_once_with(
        query="Check the document.",
        thread_id="thread-123",
        user_id=42,
        k=settings.rag_retrieval_top_k,
    )


# =========================================================
# rerank_node
# =========================================================


def test_rerank_node_builds_context_from_reranked_candidates(
    mocker,
):
    candidates = [
        {
            "chunk_id": 1,
            "document_id": 10,
            "content": "Less relevant",
            "source": "project.pdf",
            "stored_name": "project.pdf",
            "page": 1,
            "chunk_index": 1,
            "distance": 0.10,
        },
        {
            "chunk_id": 2,
            "document_id": 10,
            "content": "Project deadline is 30 June.",
            "source": "project.pdf",
            "stored_name": "project.pdf",
            "page": 2,
            "chunk_index": 2,
            "distance": 0.20,
        },
    ]

    reranked = [
        {
            **candidates[1],
            "rerank_score": 0.95,
        }
    ]

    relevant = [
        reranked[0]
    ]

    mock_filter = mocker.patch(
        "nodes.filter_relevant_candidates",
        return_value=relevant,
    )

    llm = mocker.Mock()

    mock_rerank = mocker.patch(
        "nodes.rerank_candidates",
        return_value=reranked,
    )

    mock_answerability = mocker.patch(
        "nodes.assess_query_answerability",
        return_value=True,
    )

    mock_build_context = mocker.patch(
        "nodes.build_rag_context",
        return_value={
            "context": (
                "[Source: project.pdf, page: 2]\n"
                "Project deadline is 30 June."
            ),
            "sources": [
                {
                    "source": "project.pdf",
                    "page": 2,
                    "chunk_index": 2,
                }
            ],
        },
    )

    state = {
        "messages": [
            HumanMessage(
                content="A jaki był jego termin?"
            )
        ],
        "rag_query": (
            "Jaki termin realizacji projektu Atlas?"
        ),
        "rag_candidates": candidates,
    }

    result = rerank_node(
        state=state,
        llm=llm,
    )

    mock_rerank.assert_called_once_with(
        query=(
            "Jaki termin realizacji projektu Atlas?"
        ),
        candidates=candidates,
        llm=llm,
        top_k=settings.rag_final_top_k,
    )

    mock_filter.assert_called_once_with(
        candidates=reranked,
        threshold=(
            settings.rag_relevance_threshold
        ),
    )

    mock_build_context.assert_called_once_with(
        relevant,
        max_chars=(
            settings.rag_context_max_chars
        ),
    )

    mock_answerability.assert_called_once_with(
        query=(
            "Jaki termin realizacji projektu Atlas?"
        ),
        candidates=relevant,
        llm=llm,
    )


    assert result == {
        "rag_context": (
            "[Source: project.pdf, page: 2]\n"
            "Project deadline is 30 June."
        ),
        "rag_sources": [
            {
                "source": "project.pdf",
                "page": 2,
                "chunk_index": 2,
            }
        ],
    }



def test_rerank_node_removes_context_when_global_gate_rejects(
    mocker,
):
    candidates = [
        {
            "chunk_id": 1,
            "document_id": 10,
            "content": (
                "Projekt Atlas kończy się "
                "30 czerwca."
            ),
            "source": "atlas.pdf",
            "stored_name": "atlas.pdf",
            "page": 1,
            "chunk_index": 1,
            "distance": 0.10,
        }
    ]

    reranked = [
        {
            **candidates[0],
            "rerank_score": 1.0,
            "rerank_answerable": True,
        }
    ]

    mocker.patch(
        "nodes.rerank_candidates",
        return_value=reranked,
    )

    mocker.patch(
        "nodes.filter_relevant_candidates",
        return_value=reranked,
    )

    mocker.patch(
        "nodes.assess_query_answerability",
        return_value=False,
    )

    mock_build_context = mocker.patch(
        "nodes.build_rag_context",
        return_value={
            "context": "",
            "sources": [],
        },
    )

    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Czy projekt Atlas "
                    "jest opóźniony?"
                )
            )
        ],
        "rag_query": (
            "Czy projekt Atlas jest opóźniony?"
        ),
        "rag_candidates": candidates,
    }

    result = rerank_node(
        state=state,
        llm=mocker.Mock(),
    )

    mock_build_context.assert_called_once_with(
        [],
        max_chars=(
            settings.rag_context_max_chars
        ),
    )

    assert result == {
        "rag_context": "",
        "rag_sources": [],
    }

    assert (
        route_after_rerank(
            {
                **state,
                **result,
            }
        )
        == "no_answer"
    )




# =========================================================
# route_after_rerank
# =========================================================


def test_route_after_rerank_routes_to_generate_when_context_exists():
    state = {
        "messages": [],
        "rag_context": (
            "Relevant document context."
        ),
    }

    assert (
        route_after_rerank(state)
        == "generate"
    )






def test_route_after_rerank_routes_to_no_answer_when_context_empty():
    state = {
        "messages": [],
        "rag_context": "",
    }

    assert (
        route_after_rerank(state)
        == "no_answer"
    )



# =========================================================
# no_answer_node
# =========================================================


@pytest.mark.parametrize(
    (
        "language",
        "expected_fragment",
    ),
    [
        (
            "pl-PL",
            "Nie znalazłem wystarczająco",
        ),
        (
            "en-US",
            "I couldn't find enough relevant",
        ),
    ],
)




def test_no_answer_node_returns_safe_message(
    language,
    expected_fragment,
):
    state = {
        "messages": [
            HumanMessage(
                content="Question"
            )
        ],
        "speech_language": language,
    }

    result = no_answer_node(
        state
    )

    assert len(
        result["messages"]
    ) == 1

    message = result[
        "messages"
    ][0]

    assert isinstance(
        message,
        AIMessage,
    )

    assert (
        expected_fragment
        in message.content
    )




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




def test_generate_rag_node_uses_strict_grounding_prompt(
    mocker,
):
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Kiedy rozpoczął się projekt "
                    "Atlas i kiedy się zakończy?"
                )
            )
        ],
        "rag_context": (
            "Pilotaż rozpocznie się 15 maja. "
            "Termin zakończenia projektu Atlas "
            "to 30 czerwca."
        ),
        "rag_sources": [],
    }

    llm = mocker.Mock()

    llm.invoke.return_value = AIMessage(
        content=(
            "Dokument nie podaje daty "
            "rozpoczęcia projektu. "
            "Termin zakończenia to 30 czerwca."
        )
    )

    writer = mocker.Mock()

    generate_rag_node(
        state=state,
        llm=llm,
        writer=writer,
    )

    messages = (
        llm.invoke
        .call_args
        .args[0]
    )

    system_prompt = (
        messages[0].content
    )

    normalized_prompt = " ".join(
        system_prompt.split()
    )

    assert (
        "Do not infer, assume, guess"
        in normalized_prompt
    )

    assert (
        "A project deadline does not imply "
        "when the project started."
        in normalized_prompt
    )

    assert (
        "answers only some of them"
        in normalized_prompt
    )


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