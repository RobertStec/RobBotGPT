from unittest.mock import Mock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

import nodes





def test_advanced_rag_flow_routes_relevant_context_to_generate(
    mocker,
):
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Jaki był termin projektu Atlas?"
                )
            )
        ],
        "rag_query": (
            "Jaki był termin projektu Atlas?"
        ),
        "rag_candidates": [
            {
                "chunk_id": 1,
                "document_id": 10,
                "content": (
                    "Termin projektu Atlas "
                    "to 30 czerwca."
                ),
                "source": "atlas.pdf",
                "stored_name": "atlas.pdf",
                "page": 2,
                "chunk_index": 3,
                "distance": 0.12,
            }
        ],
    }

    reranked = [
        {
            **state["rag_candidates"][0],
            "rerank_score": 0.95,
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

    llm = Mock()

    result = nodes.rerank_node(
        state=state,
        llm=llm,
    )

    next_node = nodes.route_after_rerank(
        {
            **state,
            **result,
        }
    )

    assert next_node == "generate"

    assert (
        "Termin projektu Atlas"
        in result["rag_context"]
    )

    assert result["rag_sources"] == [
        {
            "source": "atlas.pdf",
            "page": 2,
            "chunk_index": 3,
        }
    ]






def test_advanced_rag_flow_routes_empty_context_to_no_answer(
    mocker,
):
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Jaki jest numer telefonu "
                    "prezesa?"
                )
            )
        ],
        "rag_query": (
            "Numer telefonu prezesa"
        ),
        "rag_candidates": [
            {
                "chunk_id": 1,
                "document_id": 10,
                "content": (
                    "Dokument opisuje harmonogram."
                ),
                "source": "project.pdf",
                "stored_name": "project.pdf",
                "page": 1,
                "chunk_index": 1,
                "distance": 0.30,
            }
        ],
    }

    reranked = [
        {
            **state["rag_candidates"][0],
            "rerank_score": 0.20,
        }
    ]

    mocker.patch(
        "nodes.rerank_candidates",
        return_value=reranked,
    )

    mocker.patch(
        "nodes.filter_relevant_candidates",
        return_value=[],
    )

    llm = Mock()

    result = nodes.rerank_node(
        state=state,
        llm=llm,
    )

    next_node = nodes.route_after_rerank(
        {
            **state,
            **result,
        }
    )

    assert next_node == "no_answer"

    assert result["rag_context"] == ""
    assert result["rag_sources"] == []




# Test sprawdza prawdziwe połączenie

def test_advanced_rag_flow_uses_rewritten_query_for_retrieval(
    mocker,
):
    rewrite_llm = mocker.Mock()

    rewrite_llm.invoke.return_value = AIMessage(
        content=(
            "Jaki termin realizacji projektu "
            "Atlas podano w dokumencie?"
        )
    )

    initial_state = {
        "messages": [
            HumanMessage(
                content=(
                    "W dokumencie opisano "
                    "projekt Atlas."
                )
            ),
            AIMessage(
                content=(
                    "Tak, dokument opisuje "
                    "projekt Atlas."
                )
            ),
            HumanMessage(
                content="A jaki był jego termin?"
            ),
        ]
    }

    rewrite_result = nodes.rewrite_query_node(
        state=initial_state,
        llm=rewrite_llm,
    )

    state = {
        **initial_state,
        **rewrite_result,
    }

    mock_retrieve = mocker.patch(
        "nodes.retrieve_candidates_from_rag",
        return_value=[],
    )

    writer = mocker.Mock()

    nodes.retrieve_node(
        state=state,
        config={
            "configurable": {
                "thread_id": "thread-123"
            }
        },
        writer=writer,
    )

    mock_retrieve.assert_called_once_with(
        query=(
            "Jaki termin realizacji projektu "
            "Atlas podano w dokumencie?"
        ),
        thread_id="thread-123",
        k=nodes.settings.rag_retrieval_top_k,
    )




# Test brak wyników z pgvector nie powoduje błędu i kończy się no_answer

def test_advanced_rag_flow_handles_empty_retrieval(
    mocker,
):
    mocker.patch(
        "nodes.retrieve_candidates_from_rag",
        return_value=[],
    )

    writer = mocker.Mock()

    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Co dokument mówi "
                    "o projekcie Orion?"
                )
            )
        ],
        "rag_query": (
            "Projekt Orion"
        ),
    }

    retrieve_result = nodes.retrieve_node(
        state=state,
        config={
            "configurable": {
                "thread_id": "thread-empty"
            }
        },
        writer=writer,
    )

    assert retrieve_result == {
        "rag_candidates": []
    }

    rerank_state = {
        **state,
        **retrieve_result,
    }

    result = nodes.rerank_node(
        state=rerank_state,
        llm=mocker.Mock(),
    )

    assert result == {
        "rag_context": "",
        "rag_sources": [],
    }

    assert (
        nodes.route_after_rerank(
            {
                **rerank_state,
                **result,
            }
        )
        == "no_answer"
    )




# Test fallback pgvector po awarii rerankera nadal pozwala odpowiedzieć

def test_advanced_rag_flow_preserves_pgvector_fallback(
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
            "page": 2,
            "chunk_index": 3,
            "distance": 0.10,
        }
    ]

    mocker.patch(
        "nodes.rerank_candidates",
        return_value=candidates,
    )

    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Kiedy kończy się projekt Atlas?"
                )
            )
        ],
        "rag_query": (
            "Termin zakończenia projektu Atlas"
        ),
        "rag_candidates": candidates,
    }

    result = nodes.rerank_node(
        state=state,
        llm=mocker.Mock(),
    )

    assert (
        "Projekt Atlas kończy się"
        in result["rag_context"]
    )

    assert result["rag_sources"] == [
        {
            "source": "atlas.pdf",
            "page": 2,
            "chunk_index": 3,
        }
    ]

    assert (
        nodes.route_after_rerank(
            {
                **state,
                **result,
            }
        )
        == "generate"
    )




# Test Context Builder respektuje limit również wewnątrz całego rerank_node

def test_advanced_rag_flow_respects_context_limit(
    mocker,
    monkeypatch,
):
    candidates = [
        {
            "chunk_id": 1,
            "document_id": 10,
            "content": "A" * 300,
            "source": "first.pdf",
            "stored_name": "first.pdf",
            "page": 1,
            "chunk_index": 1,
            "distance": 0.10,
            "rerank_score": 0.95,
        },
        {
            "chunk_id": 2,
            "document_id": 20,
            "content": "B" * 300,
            "source": "second.pdf",
            "stored_name": "second.pdf",
            "page": 2,
            "chunk_index": 1,
            "distance": 0.20,
            "rerank_score": 0.90,
        },
    ]

    monkeypatch.setattr(
        nodes.settings,
        "rag_context_max_chars",
        150,
    )

    mocker.patch(
        "nodes.rerank_candidates",
        return_value=candidates,
    )

    mocker.patch(
        "nodes.filter_relevant_candidates",
        return_value=candidates,
    )

    state = {
        "messages": [
            HumanMessage(
                content="Question"
            )
        ],
        "rag_query": "Question",
        "rag_candidates": candidates,
    }

    result = nodes.rerank_node(
        state=state,
        llm=mocker.Mock(),
    )

    assert len(
        result["rag_context"]
    ) <= 150

    assert result["rag_sources"] == [
        {
            "source": "first.pdf",
            "page": 1,
            "chunk_index": 1,
        }
    ]
