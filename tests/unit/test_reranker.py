from types import SimpleNamespace

import pytest

from reranker import (
    assess_query_answerability,
    filter_relevant_candidates,
    rerank_candidates,
)




def make_candidate(
    chunk_id: int,
    content: str,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": 1,
        "content": content,
        "source": "manual.pdf",
        "stored_name": "manual.pdf",
        "page": 1,
        "chunk_index": chunk_id,
        "distance": chunk_id / 10,
    }


def test_rerank_candidates_reorders_by_relevance(
    mocker,
):
    candidates = [
        make_candidate(
            1,
            "General information.",
        ),
        make_candidate(
            2,
            "Project deadline is 30 June.",
        ),
        make_candidate(
            3,
            "Contact information.",
        ),
    ]

    llm = mocker.Mock()

    structured_llm = mocker.Mock()

    llm.with_structured_output.return_value = (
        structured_llm
    )

    structured_llm.invoke.return_value = (
        SimpleNamespace(
            items=[
                SimpleNamespace(
                    chunk_id=1,
                    score=0.20,
                    answerable=False,
                ),
                SimpleNamespace(
                    chunk_id=2,
                    score=0.95,
                    answerable=True,
                ),
                SimpleNamespace(
                    chunk_id=3,
                    score=0.10,
                    answerable=False,
                ),
            ]
        )
    )

    result = rerank_candidates(
        query="Jaki jest termin projektu?",
        candidates=candidates,
        llm=llm,
        top_k=2,
    )

    assert [
        candidate["chunk_id"]
        for candidate in result
    ] == [
        2,
        1,
    ]

    assert result[0][
        "rerank_score"
    ] == pytest.approx(
        0.95
    )

    assert (
        result[0]["rerank_answerable"]
        is True
    )




def test_rerank_candidates_returns_original_order_on_llm_error(
    mocker,
):
    candidates = [
        make_candidate(1, "First"),
        make_candidate(2, "Second"),
        make_candidate(3, "Third"),
    ]

    llm = mocker.Mock()

    structured_llm = mocker.Mock()

    llm.with_structured_output.return_value = (
        structured_llm
    )

    structured_llm.invoke.side_effect = (
        RuntimeError(
            "Reranker unavailable"
        )
    )

    result = rerank_candidates(
        query="question",
        candidates=candidates,
        llm=llm,
        top_k=2,
    )

    assert result == candidates[:2]




def test_rerank_candidates_handles_empty_candidates(
    mocker,
):
    llm = mocker.Mock()

    result = rerank_candidates(
        query="question",
        candidates=[],
        llm=llm,
        top_k=4,
    )

    assert result == []

    llm.with_structured_output.assert_not_called()




def test_filter_relevant_candidates_removes_low_scores():
    candidates = [
        {
            "chunk_id": 1,
            "rerank_score": 0.91,
        },
        {
            "chunk_id": 2,
            "rerank_score": 0.55,
        },
        {
            "chunk_id": 3,
            "rerank_score": 0.31,
        },
    ]

    result = filter_relevant_candidates(
        candidates=candidates,
        threshold=0.55,
    )

    assert [
        candidate["chunk_id"]
        for candidate in result
    ] == [1, 2]





def test_filter_relevant_candidates_returns_empty_when_none_relevant():
    candidates = [
        {
            "chunk_id": 1,
            "rerank_score": 0.20,
        },
        {
            "chunk_id": 2,
            "rerank_score": 0.40,
        },
    ]

    result = filter_relevant_candidates(
        candidates=candidates,
        threshold=0.55,
    )

    assert result == []





def test_filter_relevant_candidates_preserves_pgvector_fallback():
    candidates = [
        {
            "chunk_id": 1,
            "distance": 0.15,
        },
        {
            "chunk_id": 2,
            "distance": 0.22,
        },
    ]

    result = filter_relevant_candidates(
        candidates=candidates,
        threshold=0.55,
    )

    assert result == candidates





def test_filter_relevant_candidates_rejects_not_answerable():
    candidates = [
        {
            "chunk_id": 1,
            "rerank_score": 0.95,
            "rerank_answerable": False,
        }
    ]

    result = filter_relevant_candidates(
        candidates=candidates,
        threshold=0.55,
    )

    assert result == []





def test_filter_relevant_candidates_keeps_answerable_candidate():
    candidates = [
        {
            "chunk_id": 1,
            "rerank_score": 0.80,
            "rerank_answerable": True,
        }
    ]

    result = filter_relevant_candidates(
        candidates=candidates,
        threshold=0.55,
    )

    assert result == candidates




def test_assess_query_answerability_rejects_unsupported_inference(
    mocker,
):
    candidates = [
        make_candidate(
            1,
            (
                "Projekt Atlas ma termin zakończenia "
                "30 czerwca. Podczas prac omawiane są "
                "problemy techniczne."
            ),
        )
    ]

    llm = mocker.Mock()
    structured_llm = mocker.Mock()

    llm.with_structured_output.return_value = (
        structured_llm
    )

    structured_llm.invoke.return_value = (
        SimpleNamespace(
            answerable=False,
            confidence=0.95,
            reasoning=(
                "The evidence does not state "
                "that the project is delayed."
            ),
        )
    )

    result = assess_query_answerability(
        query="Czy projekt Atlas jest opóźniony?",
        candidates=candidates,
        llm=llm,
    )

    assert result is False

    structured_llm.invoke.assert_called_once()






def test_assess_query_answerability_accepts_partial_answer(
    mocker,
):
    candidates = [
        make_candidate(
            1,
            (
                "Termin zakończenia projektu Atlas "
                "to 30 czerwca."
            ),
        )
    ]

    llm = mocker.Mock()
    structured_llm = mocker.Mock()

    llm.with_structured_output.return_value = (
        structured_llm
    )

    structured_llm.invoke.return_value = (
        SimpleNamespace(
            answerable=True,
            confidence=0.90,
            reasoning=(
                "The evidence answers the project "
                "end-date part of the question."
            ),
        )
    )

    result = assess_query_answerability(
        query=(
            "Kiedy rozpoczął się projekt Atlas "
            "i kiedy się zakończy?"
        ),
        candidates=candidates,
        llm=llm,
    )

    assert result is True





def test_assess_query_answerability_rejects_empty_candidates(
    mocker,
):
    llm = mocker.Mock()

    result = assess_query_answerability(
        query="Question",
        candidates=[],
        llm=llm,
    )

    assert result is False

    llm.with_structured_output.assert_not_called()