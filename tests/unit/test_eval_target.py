from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from evals.target import (
    build_eval_messages,
    build_rag_eval_target,
)





def test_build_eval_messages_builds_history_and_question():
    inputs = {
        "history": [
            {
                "role": "user",
                "content": "Projekt Atlas.",
            },
            {
                "role": "assistant",
                "content": "Rozumiem.",
            },
        ],
        "question": "Jaki był termin?",
    }

    messages = build_eval_messages(
        inputs
    )

    assert len(messages) == 3

    assert isinstance(
        messages[0],
        HumanMessage,
    )

    assert isinstance(
        messages[1],
        AIMessage,
    )

    assert isinstance(
        messages[2],
        HumanMessage,
    )

    assert (
        messages[-1].content
        == "Jaki był termin?"
    )





def test_rag_eval_target_returns_generate_result(
    mocker,
):
    llm = mocker.Mock()

    mocker.patch(
        "evals.target.rewrite_query_node",
        return_value={
            "rag_query": (
                "Termin projektu Atlas"
            )
        },
    )

    mocker.patch(
        "evals.target.retrieve_node",
        return_value={
            "rag_candidates": [
                {
                    "chunk_id": 1
                }
            ]
        },
    )

    mocker.patch(
        "evals.target.rerank_node",
        return_value={
            "rag_context": (
                "Termin projektu: "
                "30 czerwca."
            ),
            "rag_sources": [
                {
                    "source": "atlas.pdf",
                    "page": 2,
                    "chunk_index": 1,
                }
            ],
        },
    )

    mocker.patch(
        "evals.target.route_after_rerank",
        return_value="generate",
    )

    mocker.patch(
        "evals.target.generate_rag_node",
        return_value={
            "messages": [
                AIMessage(
                    content=(
                        "Termin to 30 czerwca."
                    )
                )
            ]
        },
    )

    mock_eval_user = mocker.Mock()
    mock_eval_user.id = 101

    mocker.patch(
        "evals.target.get_eval_user",
        return_value=mock_eval_user,
    )

    target = build_rag_eval_target(
        thread_id="eval-thread",
        llm=llm,
    )

    result = target({
        "question": (
            "Jaki jest termin "
            "projektu Atlas?"
        ),
        "history": [],
    })

    assert result == {
        "answer": (
            "Termin to 30 czerwca."
        ),
        "route": "generate",
        "sources": [
            {
                "source": "atlas.pdf",
                "page": 2,
                "chunk_index": 1,
            }
        ],
        "rewritten_query": (
            "Termin projektu Atlas"
        ),
    }





def test_rag_eval_target_returns_no_answer_result(
    mocker,
):
    llm = mocker.Mock()

    mocker.patch(
        "evals.target.rewrite_query_node",
        return_value={
            "rag_query": (
                "Numer telefonu prezesa"
            )
        },
    )

    mocker.patch(
        "evals.target.retrieve_node",
        return_value={
            "rag_candidates": []
        },
    )

    mocker.patch(
        "evals.target.rerank_node",
        return_value={
            "rag_context": "",
            "rag_sources": [],
        },
    )

    mocker.patch(
        "evals.target.route_after_rerank",
        return_value="no_answer",
    )

    mocker.patch(
        "evals.target.no_answer_node",
        return_value={
            "messages": [
                AIMessage(
                    content=(
                        "Nie znalazłem "
                        "wystarczająco istotnych "
                        "informacji."
                    )
                )
            ]
        },
    )

    mock_eval_user = mocker.Mock()
    mock_eval_user.id = 101

    mocker.patch(
        "evals.target.get_eval_user",
        return_value=mock_eval_user,
    )

    target = build_rag_eval_target(
        thread_id="eval-thread",
        llm=llm,
    )

    result = target({
        "question": (
            "Jaki jest numer telefonu "
            "prezesa?"
        ),
        "history": [],
    })

    assert (
        result["route"]
        == "no_answer"
    )

    assert result["sources"] == []

    assert (
        result["rewritten_query"]
        == "Numer telefonu prezesa"
    )





def test_rag_eval_target_rejects_empty_thread_id():
    try:
        build_rag_eval_target(
            thread_id="   ",
        )

    except ValueError as exc:
        assert (
            "thread_id cannot be empty"
            in str(exc)
        )

    else:
        raise AssertionError(
            "ValueError was not raised."
        )