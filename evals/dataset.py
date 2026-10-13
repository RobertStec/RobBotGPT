import json
from pathlib import Path

from pydantic import TypeAdapter

from evals.schemas import RagEvalCase


DEFAULT_RAG_EVAL_PATH = (
    Path(__file__).parent
    / "data"
    / "rag_eval_cases.json"
)


def load_rag_eval_cases(
    path: str | Path | None = None,
) -> list[RagEvalCase]:
    """
    Load and validate the local RAG evaluation dataset.
    """

    dataset_path = (
        Path(path)
        if path is not None
        else DEFAULT_RAG_EVAL_PATH
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{dataset_path}"
        )

    try:
        raw_data = json.loads(
            dataset_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Evaluation dataset contains "
            "invalid JSON."
        ) from exc

    if not isinstance(
        raw_data,
        list,
    ):
        raise ValueError(
            "Evaluation dataset must "
            "contain a JSON list."
        )

    if not raw_data:
        raise ValueError(
            "Evaluation dataset cannot be empty."
        )

    adapter = TypeAdapter(
        list[RagEvalCase]
    )

    cases = adapter.validate_python(
        raw_data
    )

    case_ids = [
        case.id
        for case in cases
    ]

    if (
        len(case_ids)
        != len(set(case_ids))
    ):
        raise ValueError(
            "Evaluation case IDs "
            "must be unique."
        )

    return cases