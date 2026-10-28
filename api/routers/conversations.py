from fastapi import (
    APIRouter,
    Depends,
)

from api.dependencies.auth import (
    get_current_user,
)

from services.conversation_service import (
    delete_conversation_resources,
    get_conversation_history,
    list_conversation_summaries,
)


router = APIRouter()


@router.get("/conversations")
async def conversations(
    current_user=Depends(
        get_current_user
    ),
):
    return {
        "conversations": (
            list_conversation_summaries(
                user_id=current_user.id
            )
        )
    }


@router.get("/history/{thread_id}")
async def history(
    thread_id: str,
    current_user=Depends(
        get_current_user
    ),
):
    return {
        "messages": (
            get_conversation_history(
                thread_id,
                user_id=current_user.id,
            )
        )
    }


@router.delete(
    "/conversations/{thread_id}"
)
async def delete_conversation_endpoint(
    thread_id: str,
    current_user=Depends(
        get_current_user
    ),
):
    result = (
        delete_conversation_resources(
            thread_id,
            user_id=current_user.id,
        )
    )

    return {
        "success": True,
        **result,
    }