from core.config import settings

from evals.evaluators import (
    build_answer_correctness_evaluator,
    no_answer_correctness,
    route_correctness,
    source_correctness,
)
from evals.langsmith_client import (
    get_langsmith_client,
)
from evals.seed_corpus import (
    EVAL_THREAD_ID,
)
from evals.sync_dataset import (
    DATASET_NAME,
)
from evals.target import (
    build_rag_eval_target,
)
from evals.tracing import (
    build_eval_metadata,
)

from evals.regression import (
    RegressionFailure,
    check_experiment_regression,
)




EXPERIMENT_PREFIX = (
    "robbotgpt-rag-regression"
)



def run_rag_experiment():
    """
    Run the RobBotGPT Advanced RAG evaluation
    against the LangSmith dataset.
    """

    client = get_langsmith_client()

    target = build_rag_eval_target(
        thread_id=EVAL_THREAD_ID,
    )

    answer_correctness = (
        build_answer_correctness_evaluator()
    )

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            route_correctness,
            source_correctness,
            no_answer_correctness,
            answer_correctness,
        ],
        experiment_prefix=(
            EXPERIMENT_PREFIX
        ),
        metadata=build_eval_metadata(
            settings.openai_model
        ),
        max_concurrency=1,
    )

    return results




if __name__ == "__main__":
    results = run_rag_experiment()

    print(
        "\nExperiment completed."
    )

    try:
        regression_result = (
            check_experiment_regression(
                results
            )
        )

    except RegressionFailure as exc:
        print(
            "\nREGRESSION GATE: FAILED"
        )

        print(
            str(exc)
        )

        raise SystemExit(1)

    print(
        "\nREGRESSION GATE: PASSED"
    )

    print(
        "examples:",
        regression_result[
            "example_count"
        ],
    )

    for name, score in (
        regression_result[
            "metrics"
        ].items()
    ):
        required = (
            regression_result[
                "thresholds"
            ][name]
        )

        print(
            f"{name}: "
            f"{score:.3f} "
            f"(required >= "
            f"{required:.3f})"
        )