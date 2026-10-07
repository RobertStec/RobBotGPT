from core.exceptions import (
    AppError,
    ConversationDeletionError,
    ConversationNotFoundError,
)

from database import (
    conversation_exists,
    delete_conversation as delete_conversation_from_db,
    get_chat_history,
    list_conversations,
)

from agent import (
    delete_thread_checkpoints,
)

from rag import (
    delete_thread_documents,
)

from services.streaming import (
    parse_message_sources,
)


def list_conversation_summaries() -> list[dict]:
    """
    Return conversations serialized
    for the API layer.
    """

    items = list_conversations()

    return [
        {
            "thread_id": item.thread_id,
            "title": item.title,
            "created_at": (
                item.created_at.isoformat()
            ),
            "updated_at": (
                item.updated_at.isoformat()
            ),
        }
        for item in items
    ]


def get_conversation_history(
    thread_id: str,
) -> list[dict]:
    """
    Return serialized chat history
    for one conversation.
    """

    messages = get_chat_history(
        thread_id
    )

    return [
        {
            "role": message.role,
            "content": message.content,
            "sources": parse_message_sources(
                message.sources
            ),
        }
        for message in messages
    ]


def delete_conversation_resources(
    thread_id: str,
) -> dict:
    """
    Delete all resources belonging to a conversation.

    SQL data is deleted last so a partial failure
    can be retried safely.
    """

    try:
        if not conversation_exists(
            thread_id
        ):
            raise ConversationNotFoundError()

        # 1. RAG + uploaded files
        try:
            rag_result = (
                delete_thread_documents(
                    thread_id
                )
            )

        except Exception as exc:
            raise ConversationDeletionError(
                internal_message=(
                    "RAG cleanup failed: "
                    f"{exc}"
                )
            ) from exc

        # 2. LangGraph checkpoints
        try:
            delete_thread_checkpoints(
                thread_id
            )

        except Exception as exc:
            raise ConversationDeletionError(
                internal_message=(
                    "Checkpoint cleanup failed: "
                    f"{exc}"
                )
            ) from exc

        # 3. SQL database LAST
        try:
            deleted = (
                delete_conversation_from_db(
                    thread_id
                )
            )

        except Exception as exc:
            raise ConversationDeletionError(
                internal_message=(
                    "Database cleanup failed: "
                    f"{exc}"
                )
            ) from exc

        if not deleted:
            raise ConversationDeletionError(
                internal_message=(
                    "Conversation disappeared "
                    "before final database deletion."
                )
            )

        return {
            "message": (
                "Conversation deleted."
            ),
            "deleted_chunks": (
                rag_result[
                    "deleted_chunks"
                ]
            ),
            "deleted_files": (
                rag_result[
                    "deleted_files"
                ]
            ),
        }

    except AppError:
        raise

    except Exception as exc:
        raise ConversationDeletionError(
            internal_message=str(exc)
        ) from exc