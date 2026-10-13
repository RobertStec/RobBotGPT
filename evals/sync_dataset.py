from langsmith.utils import (
    LangSmithNotFoundError,
)

from evals.dataset import (
    load_rag_eval_cases,
)
from evals.langsmith_client import (
    get_langsmith_client,
)


DATASET_NAME = "RobBotGPT-RAG-Eval"

DATASET_DESCRIPTION = (
    "Evaluation dataset for the "
    "RobBotGPT Advanced RAG pipeline."
)


def case_to_langsmith_example(
    case,
) -> dict:
    """
    Convert a local RagEvalCase into
    a LangSmith dataset example.
    """

    return {
        "inputs": {
            "question": case.question,
            "history": [
                message.model_dump()
                for message in case.history
            ],
        },
        "outputs": {
            "reference_answer": (
                case.reference_answer
            ),
            "expected_route": (
                case.expected_route
            ),
            "expected_source": (
                case.expected_source
            ),
        },
        "metadata": {
            "case_id": case.id,
            "category": case.category,
        },
    }


def sync_rag_eval_dataset() -> dict:
    """
    Create the LangSmith dataset if needed
    and upload missing local evaluation cases.
    """

    client = get_langsmith_client()

    cases = load_rag_eval_cases()

    try:
        dataset = client.read_dataset(
            dataset_name=DATASET_NAME,
        )

        dataset_created = False

    except LangSmithNotFoundError:

        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                DATASET_DESCRIPTION
            ),
        )

        dataset_created = True

    existing_examples = list(
        client.list_examples(
            dataset_id=dataset.id,
        )
    )

    existing_case_ids = {
        example.metadata.get(
            "case_id"
        )
        for example in existing_examples
        if example.metadata
    }

    new_examples = [
        case_to_langsmith_example(
            case
        )
        for case in cases
        if case.id
        not in existing_case_ids
    ]

    if new_examples:
        client.create_examples(
            dataset_id=dataset.id,
            examples=new_examples,
        )

    return {
        "dataset_name": DATASET_NAME,
        "dataset_id": str(dataset.id),
        "dataset_created": (
            dataset_created
        ),
        "uploaded_examples": (
            len(new_examples)
        ),
        "local_examples": len(cases),
    }


if __name__ == "__main__":
    result = sync_rag_eval_dataset()

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )