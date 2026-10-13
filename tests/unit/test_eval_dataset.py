import json

import pytest
from pydantic import ValidationError

from evals.dataset import (
    load_rag_eval_cases,
)


def test_load_rag_eval_cases_loads_default_dataset():
    cases = load_rag_eval_cases()

    assert len(cases) == 20

    assert cases[0].id == "rag_fact_001"

    assert (
        cases[0].expected_route
        == "generate"
    )


def test_load_rag_eval_cases_parses_query_rewrite_history():
    cases = load_rag_eval_cases()

    case = next(
        item
        for item in cases
        if item.id == "rag_rewrite_001"
    )

    assert (
        case.category
        == "query_rewrite"
    )

    assert len(
        case.history
    ) == 2

    assert (
        case.history[0].role
        == "user"
    )


def test_load_rag_eval_cases_rejects_missing_file(
    tmp_path,
):
    missing_file = (
        tmp_path
        / "missing.json"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_rag_eval_cases(
            missing_file
        )


def test_load_rag_eval_cases_rejects_invalid_json(
    tmp_path,
):
    dataset_path = (
        tmp_path
        / "invalid.json"
    )

    dataset_path.write_text(
        "{ invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        load_rag_eval_cases(
            dataset_path
        )


def test_load_rag_eval_cases_rejects_duplicate_ids(
    tmp_path,
):
    dataset = [
        {
            "id": "duplicate",
            "category": "factual",
            "question": "Question one?",
            "expected_route": "generate",
            "reference_answer": "Answer",
            "expected_source": "file.pdf",
        },
        {
            "id": "duplicate",
            "category": "factual",
            "question": "Question two?",
            "expected_route": "generate",
            "reference_answer": "Answer",
            "expected_source": "file.pdf",
        },
    ]

    dataset_path = (
        tmp_path
        / "duplicates.json"
    )

    dataset_path.write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        load_rag_eval_cases(
            dataset_path
        )


def test_load_rag_eval_cases_rejects_generate_without_reference(
    tmp_path,
):
    dataset = [
        {
            "id": "invalid",
            "category": "factual",
            "question": "Question?",
            "expected_route": "generate",
            "reference_answer": None,
            "expected_source": "file.pdf",
        }
    ]

    dataset_path = (
        tmp_path
        / "invalid_case.json"
    )

    dataset_path.write_text(
        json.dumps(dataset),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="reference_answer",
    ):
        load_rag_eval_cases(
            dataset_path
        )