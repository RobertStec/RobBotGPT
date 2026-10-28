from pathlib import Path

from rag import (
    add_document_to_rag,
    delete_thread_documents,
)

from database import (
    create_user,
    get_user_by_email,
)




EVAL_THREAD_ID = "langsmith-rag-eval-v1"

EVAL_USER_EMAIL = (
    "langsmith-eval@robbotgpt.local"
)

EVAL_USER_PASSWORD_HASH = (
    "evaluation-only"
)

EVAL_DOCUMENT = (
    Path(__file__).parent
    / "data"
    / "atlas.txt"
)



def get_eval_user():
    user = get_user_by_email(
        EVAL_USER_EMAIL
    )

    if user is None:
        user = create_user(
            email=EVAL_USER_EMAIL,
            password_hash=(
                EVAL_USER_PASSWORD_HASH
            ),
        )

    return user




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

    user = get_eval_user()

    # Dedicated evaluation thread, so resetting it
    # does not affect normal user documents.
    delete_thread_documents(
        EVAL_THREAD_ID,
        user_id=user.id,
    )

    result = add_document_to_rag(
        file_path=str(EVAL_DOCUMENT),
        thread_id=EVAL_THREAD_ID,
        original_filename="atlas.pdf",
        user_id=user.id,
    )

    return {
        "thread_id": EVAL_THREAD_ID,
        "source": result["filename"],
        "chunks": result["chunks"],
        "user_id": user.id,
    }


if __name__ == "__main__":
    result = seed_eval_corpus()

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )
