from evals.evaluators import (
    AnswerCorrectnessGrade,
    build_answer_correctness_evaluator,
    no_answer_correctness,
    route_correctness,
    source_correctness,
)




def test_route_correctness_returns_true_for_matching_route():
    outputs = {
        "route": "generate"
    }

    reference_outputs = {
        "expected_route": "generate"
    }

    assert route_correctness(
        outputs,
        reference_outputs,
    )




def test_route_correctness_returns_false_for_wrong_route():
    outputs = {
        "route": "no_answer"
    }

    reference_outputs = {
        "expected_route": "generate"
    }

    assert not route_correctness(
        outputs,
        reference_outputs,
    )




def test_source_correctness_finds_expected_source():
    outputs = {
        "sources": [
            {
                "source": "atlas.pdf",
                "page": 2,
                "chunk_index": 1,
            }
        ]
    }

    reference_outputs = {
        "expected_source": "atlas.pdf"
    }

    assert source_correctness(
        outputs,
        reference_outputs,
    )



def test_source_correctness_rejects_wrong_source():
    outputs = {
        "sources": [
            {
                "source": "other.pdf",
                "page": 1,
                "chunk_index": 1,
            }
        ]
    }

    reference_outputs = {
        "expected_source": "atlas.pdf"
    }

    assert not source_correctness(
        outputs,
        reference_outputs,
    )




def test_source_correctness_accepts_empty_sources_for_no_answer():
    outputs = {
        "sources": []
    }

    reference_outputs = {
        "expected_source": None
    }

    assert source_correctness(
        outputs,
        reference_outputs,
    )



def test_no_answer_correctness_accepts_correct_no_answer():
    outputs = {
        "route": "no_answer",
        "sources": [],
    }

    reference_outputs = {
        "expected_route": "no_answer"
    }

    assert no_answer_correctness(
        outputs,
        reference_outputs,
    )



def test_no_answer_correctness_rejects_generate_for_no_answer_case():
    outputs = {
        "route": "generate",
        "sources": [
            {
                "source": "atlas.pdf"
            }
        ],
    }

    reference_outputs = {
        "expected_route": "no_answer"
    }

    assert not no_answer_correctness(
        outputs,
        reference_outputs,
    )



def test_no_answer_correctness_ignores_generate_cases():
    outputs = {
        "route": "generate",
        "sources": [
            {
                "source": "atlas.pdf"
            }
        ],
    }

    reference_outputs = {
        "expected_route": "generate"
    }

    assert no_answer_correctness(
        outputs,
        reference_outputs,
    )




def test_answer_correctness_uses_llm_judge(
    mocker,
):
    judge_llm = mocker.Mock()
    structured_judge = mocker.Mock()

    judge_llm.with_structured_output.return_value = (
        structured_judge
    )

    structured_judge.invoke.return_value = (
        AnswerCorrectnessGrade(
            score=0.95,
            reasoning=(
                "The generated answer matches "
                "the reference answer."
            ),
        )
    )

    evaluator = (
        build_answer_correctness_evaluator(
            judge_llm=judge_llm,
        )
    )

    result = evaluator(
        inputs={
            "question": (
                "Jaki jest termin projektu Atlas?"
            )
        },
        outputs={
            "answer": (
                "Projekt Atlas kończy się "
                "30 czerwca."
            )
        },
        reference_outputs={
            "reference_answer": (
                "Termin zakończenia projektu "
                "Atlas to 30 czerwca."
            )
        },
    )

    assert (
        result["key"]
        == "answer_correctness"
    )

    assert result["score"] == 0.95

    assert result["comment"]

    structured_judge.invoke.assert_called_once()




def test_answer_correctness_skips_no_answer_case(
    mocker,
):
    judge_llm = mocker.Mock()
    structured_judge = mocker.Mock()

    judge_llm.with_structured_output.return_value = (
        structured_judge
    )

    evaluator = (
        build_answer_correctness_evaluator(
            judge_llm=judge_llm,
        )
    )

    result = evaluator(
        inputs={
            "question": (
                "Jaki jest numer telefonu prezesa?"
            )
        },
        outputs={
            "answer": (
                "Nie znalazłem informacji."
            )
        },
        reference_outputs={
            "reference_answer": None
        },
    )

    assert (
        result["key"]
        == "answer_correctness"
    )

    assert result["score"] is None

    structured_judge.invoke.assert_not_called()




def test_answer_correctness_scores_empty_answer_as_zero(
    mocker,
):
    judge_llm = mocker.Mock()
    structured_judge = mocker.Mock()

    judge_llm.with_structured_output.return_value = (
        structured_judge
    )

    evaluator = (
        build_answer_correctness_evaluator(
            judge_llm=judge_llm,
        )
    )

    result = evaluator(
        inputs={
            "question": "Question?"
        },
        outputs={
            "answer": ""
        },
        reference_outputs={
            "reference_answer": "Answer."
        },
    )

    assert result["score"] == 0.0

    structured_judge.invoke.assert_not_called()