from evals.regression import (
    RegressionFailure,
    check_regression_metrics,
)


def main():
    metrics = {
        "route_correctness": 0.90,
        "source_correctness": 1.00,
        "no_answer_correctness": 1.00,
        "answer_correctness": 1.00,
    }

    try:
        check_regression_metrics(
            metrics=metrics,
            example_count=20,
        )

    except RegressionFailure as exc:
        print(
            "REGRESSION GATE: FAILED"
        )

        print(
            str(exc)
        )

        raise SystemExit(1)

    print(
        "REGRESSION GATE: PASSED"
    )


if __name__ == "__main__":
    main()