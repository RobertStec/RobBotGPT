from dataclasses import dataclass


class RegressionFailure(RuntimeError):
    """
    Raised when an evaluation experiment
    does not satisfy the regression policy.
    """


@dataclass(frozen=True)
class RegressionPolicy:
    route_correctness: float = 0.95
    source_correctness: float = 0.95
    no_answer_correctness: float = 0.95
    answer_correctness: float = 0.90
    min_examples: int = 20


DEFAULT_REGRESSION_POLICY = (
    RegressionPolicy()
)




def check_regression_metrics(
    metrics: dict[str, float],
    example_count: int,
    policy: RegressionPolicy = (
        DEFAULT_REGRESSION_POLICY
    ),
) -> dict:
    """
    Validate experiment metrics against
    the required quality thresholds.
    """

    if example_count < policy.min_examples:
        raise RegressionFailure(
            "Evaluation dataset is too small: "
            f"{example_count} examples, "
            f"minimum required is "
            f"{policy.min_examples}."
        )

    thresholds = {
        "route_correctness": (
            policy.route_correctness
        ),
        "source_correctness": (
            policy.source_correctness
        ),
        "no_answer_correctness": (
            policy.no_answer_correctness
        ),
        "answer_correctness": (
            policy.answer_correctness
        ),
    }

    missing_metrics = [
        name
        for name in thresholds
        if name not in metrics
    ]

    if missing_metrics:
        raise RegressionFailure(
            "Missing regression metrics: "
            + ", ".join(missing_metrics)
        )

    failures = {}

    for name, threshold in thresholds.items():
        score = metrics[name]

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Metric {name} must be between "
                f"0.0 and 1.0."
            )

        if score < threshold:
            failures[name] = {
                "score": score,
                "required": threshold,
            }

    if failures:
        details = "; ".join(
            (
                f"{name}="
                f"{values['score']:.3f} "
                f"< {values['required']:.3f}"
            )
            for name, values
            in failures.items()
        )

        raise RegressionFailure(
            "RAG regression detected: "
            f"{details}"
        )

    return {
        "passed": True,
        "example_count": example_count,
        "metrics": metrics,
        "thresholds": thresholds,
    }





REGRESSION_METRICS = (
    "route_correctness",
    "source_correctness",
    "no_answer_correctness",
    "answer_correctness",
)



def aggregate_experiment_metrics(
    experiment_results,
) -> dict:
    """
    Aggregate row-level LangSmith evaluator scores
    into mean regression metrics.
    """

    rows = list(experiment_results)

    metric_scores = {
        name: []
        for name in REGRESSION_METRICS
    }

    for row in rows:
        evaluation_results = row.get(
            "evaluation_results",
            {},
        )

        results = evaluation_results.get(
            "results",
            [],
        )

        for result in results:
            key = getattr(
                result,
                "key",
                None,
            )

            score = getattr(
                result,
                "score",
                None,
            )

            if (
                key not in metric_scores
                or score is None
            ):
                continue

            metric_scores[key].append(
                float(score)
            )

    metrics = {}

    for name, scores in metric_scores.items():
        if not scores:
            continue

        metrics[name] = (
            sum(scores)
            / len(scores)
        )

    return {
        "metrics": metrics,
        "example_count": len(rows),
    }




def check_experiment_regression(
    experiment_results,
    policy: RegressionPolicy = (
        DEFAULT_REGRESSION_POLICY
    ),
) -> dict:
    """
    Aggregate a LangSmith experiment and apply
    the configured regression quality gate.
    """

    aggregated = (
        aggregate_experiment_metrics(
            experiment_results
        )
    )

    return check_regression_metrics(
        metrics=aggregated["metrics"],
        example_count=(
            aggregated["example_count"]
        ),
        policy=policy,
    )