from pathlib import Path

from rag import (
    add_document_to_rag,
    delete_thread_documents,
)




EVAL_THREAD_ID = "langsmith-rag-eval-v1"

EVAL_DOCUMENT = (
    Path(__file__).parent
    / "data"
    / "atlas.txt"
)



def seed_eval_corpus() -> dict:
    """
    Reset and seed the deterministic RAG corpus
    used by LangSmith evaluations.
    """

    if not EVAL_DOCUMENT.exists():
        raise FileNotFoundError(
            f"Evaluation document not found: "
            f"{EVAL_DOCUMENT}"
        )

    # Dedicated evaluation thread, so resetting it
    # does not affect normal user documents.
    delete_thread_documents(
        EVAL_THREAD_ID
    )

    result = add_document_to_rag(
        file_path=str(EVAL_DOCUMENT),
        thread_id=EVAL_THREAD_ID,
        original_filename="atlas.pdf",
    )

    return {
        "thread_id": EVAL_THREAD_ID,
        "source": result["filename"],
        "chunks": result["chunks"],
    }


if __name__ == "__main__":
    result = seed_eval_corpus()

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )