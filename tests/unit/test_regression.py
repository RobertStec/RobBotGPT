import pytest
from types import SimpleNamespace

from evals.regression import (
    RegressionFailure,
    RegressionPolicy,
    aggregate_experiment_metrics,
    check_experiment_regression,
    check_regression_metrics,
)


def test_regression_passes_when_metrics_meet_policy():
    metrics = {
        "route_correctness": 1.00,
        "source_correctness": 1.00,
        "no_answer_correctness": 1.00,
        "answer_correctness": 1.00,
    }

    result = check_regression_metrics(
        metrics=metrics,
        example_count=20,
    )

    assert result["passed"] is True
    assert result["example_count"] == 20


def test_regression_accepts_exact_threshold_values():
    metrics = {
        "route_correctness": 0.95,
        "source_correctness": 0.95,
        "no_answer_correctness": 0.95,
        "answer_correctness": 0.90,
    }

    result = check_regression_metrics(
        metrics=metrics,
        example_count=20,
    )

    assert result["passed"] is True


def test_regression_fails_when_metric_drops_below_threshold():
    metrics = {
        "route_correctness": 0.90,
        "source_correctness": 1.00,
        "no_answer_correctness": 1.00,
        "answer_correctness": 1.00,
    }

    with pytest.raises(
        RegressionFailure,
        match="route_correctness",
    ):
        check_regression_metrics(
            metrics=metrics,
            example_count=20,
        )


def test_regression_fails_when_metric_is_missing():
    metrics = {
        "route_correctness": 1.00,
        "source_correctness": 1.00,
        "answer_correctness": 1.00,
    }

    with pytest.raises(
        RegressionFailure,
        match="no_answer_correctness",
    ):
        check_regression_metrics(
            metrics=metrics,
            example_count=20,
        )


def test_regression_fails_when_dataset_is_too_small():
    metrics = {
        "route_correctness": 1.00,
        "source_correctness": 1.00,
        "no_answer_correctness": 1.00,
        "answer_correctness": 1.00,
    }

    with pytest.raises(
        RegressionFailure,
        match="too small",
    ):
        check_regression_metrics(
            metrics=metrics,
            example_count=19,
        )


def test_regression_rejects_invalid_metric_value():
    metrics = {
        "route_correctness": 1.20,
        "source_correctness": 1.00,
        "no_answer_correctness": 1.00,
        "answer_correctness": 1.00,
    }

    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        check_regression_metrics(
            metrics=metrics,
            example_count=20,
        )


def test_regression_supports_custom_policy():
    policy = RegressionPolicy(
        route_correctness=1.00,
        source_correctness=1.00,
        no_answer_correctness=1.00,
        answer_correctness=1.00,
        min_examples=10,
    )

    metrics = {
        "route_correctness": 1.00,
        "source_correctness": 1.00,
        "no_answer_correctness": 1.00,
        "answer_correctness": 1.00,
    }

    result = check_regression_metrics(
        metrics=metrics,
        example_count=10,
        policy=policy,
    )

    assert result["passed"] is True




def make_experiment_row(
    scores: dict,
):
    return {
        "evaluation_results": {
            "results": [
                SimpleNamespace(
                    key=key,
                    score=score,
                )
                for key, score
                in scores.items()
            ]
        }
    }




def test_aggregate_experiment_metrics_calculates_means():
    rows = [
        make_experiment_row({
            "route_correctness": 1.0,
            "source_correctness": 1.0,
            "no_answer_correctness": 1.0,
            "answer_correctness": 1.0,
        }),
        make_experiment_row({
            "route_correctness": 0.0,
            "source_correctness": 1.0,
            "no_answer_correctness": 1.0,
            "answer_correctness": 0.5,
        }),
    ]

    result = aggregate_experiment_metrics(
        rows
    )

    assert result["example_count"] == 2

    assert (
        result["metrics"]["route_correctness"]
        == 0.5
    )

    assert (
        result["metrics"]["source_correctness"]
        == 1.0
    )

    assert (
        result["metrics"]["answer_correctness"]
        == 0.75
    )




def test_aggregate_experiment_metrics_ignores_none_scores():
    rows = [
        make_experiment_row({
            "route_correctness": 1.0,
            "source_correctness": 1.0,
            "no_answer_correctness": 1.0,
            "answer_correctness": None,
        }),
        make_experiment_row({
            "route_correctness": 1.0,
            "source_correctness": 1.0,
            "no_answer_correctness": 1.0,
            "answer_correctness": 1.0,
        }),
    ]

    result = aggregate_experiment_metrics(
        rows
    )

    assert (
        result["metrics"][
            "answer_correctness"
        ]
        == 1.0
    )




def test_check_experiment_regression_passes_good_experiment():
    scores = {
        "route_correctness": 1.0,
        "source_correctness": 1.0,
        "no_answer_correctness": 1.0,
        "answer_correctness": 1.0,
    }

    rows = [
        make_experiment_row(scores)
        for _ in range(20)
    ]

    result = check_experiment_regression(
        rows
    )

    assert result["passed"] is True




def test_check_experiment_regression_fails_bad_experiment():
    good_scores = {
        "route_correctness": 1.0,
        "source_correctness": 1.0,
        "no_answer_correctness": 1.0,
        "answer_correctness": 1.0,
    }

    bad_scores = {
        **good_scores,
        "route_correctness": 0.0,
    }

    rows = [
        make_experiment_row(
            bad_scores
            if index < 2
            else good_scores
        )
        for index in range(20)
    ]

    with pytest.raises(
        RegressionFailure,
        match="route_correctness",
    ):
        check_experiment_regression(
            rows
        )