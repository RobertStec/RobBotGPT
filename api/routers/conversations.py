from fastapi import APIRouter

from services.conversation_service import (
    delete_conversation_resources,
    get_conversation_history,
    list_conversation_summaries,
)


router = APIRouter()


@router.get("/conversations")
async def conversations():
    return {
        "conversations": (
            list_conversation_summaries()
        )
    }


@router.get("/history/{thread_id}")
async def history(
    thread_id: str,
):
    return {
        "messages": (
            get_conversation_history(
                thread_id
            )
        )
    }


@router.delete(
    "/conversations/{thread_id}"
)
async def delete_conversation_endpoint(
    thread_id: str,
):
    result = (
        delete_conversation_resources(
            thread_id
        )
    )

    return {
        "success": True,
        **result,
    }