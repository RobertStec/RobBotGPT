from langchain_core.messages import AIMessage

from evals.dataset import load_rag_eval_cases
from evals.target import build_rag_eval_target
from evals.tracing import build_eval_trace_config






def test_evaluation_pipeline_smoke(
    mocker,
):
    cases = load_rag_eval_cases()

    assert cases

    case = next(
        item
        for item in cases
        if item.expected_route == "generate"
    )

    llm = mocker.Mock()

    mocker.patch(
        "evals.target.rewrite_query_node",
        return_value={
            "rag_query": case.question
        },
    )

    mocker.patch(
        "evals.target.retrieve_node",
        return_value={
            "rag_candidates": [
                {
                    "chunk_id": 1,
                    "document_id": 10,
                    "content": (
                        case.reference_answer
                        or ""
                    ),
                    "source": (
                        case.expected_source
                        or "document.pdf"
                    ),
                    "stored_name": (
                        case.expected_source
                        or "document.pdf"
                    ),
                    "page": 1,
                    "chunk_index": 1,
                    "distance": 0.10,
                }
            ]
        },
    )

    mocker.patch(
        "evals.target.rerank_node",
        return_value={
            "rag_context": (
                case.reference_answer
                or ""
            ),
            "rag_sources": [
                {
                    "source": (
                        case.expected_source
                        or "document.pdf"
                    ),
                    "page": 1,
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
                        case.reference_answer
                        or ""
                    )
                )
            ]
        },
    )

    target = build_rag_eval_target(
        thread_id="eval-smoke-thread",
        llm=llm,
    )

    result = target({
        "question": case.question,
        "history": [
            message.model_dump()
            for message in case.history
        ],
    })

    trace_config = (
        build_eval_trace_config(
            "gpt-4o-mini"
        )
    )

    assert result["route"] == (
        case.expected_route
    )

    assert result["answer"]

    assert (
        result["sources"][0]["source"]
        == case.expected_source
    )

    assert result["rewritten_query"]

    assert "rag" in trace_config["tags"]

    assert (
        trace_config["metadata"][
            "component"
        ]
        == "advanced_rag"
    )