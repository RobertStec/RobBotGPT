from fastapi import (
    APIRouter,
    Depends,
    Request,
)

from api.dependencies.auth import (
    get_current_user,
)

from services.conversation_service import (
    delete_conversation_resources,
    get_conversation_history,
    list_conversation_summaries,
)

import core.rate_limit as rate_limit



router = APIRouter()


@router.get("/conversations")
async def conversations(
    request: Request,
    current_user=Depends(
        get_current_user
    ),
):
    identifier = (
        rate_limit
        .build_rate_limit_identifier(
            request=request,
            user_id=current_user.id,
        )
    )

    await rate_limit.enforce_rate_limit(
        scope="conversations_list",
        identifier=identifier,
        limit=60,
        window_seconds=60,
    )

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
    request: Request,
    current_user=Depends(
        get_current_user
    ),
):
    identifier = (
        rate_limit
        .build_rate_limit_identifier(
            request=request,
            user_id=current_user.id,
        )
    )

    await rate_limit.enforce_rate_limit(
        scope="conversation_history",
        identifier=identifier,
        limit=60,
        window_seconds=60,
    )

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
    request: Request,
    current_user=Depends(
        get_current_user
    ),
):
    identifier = (
        rate_limit
        .build_rate_limit_identifier(
            request=request,
            user_id=current_user.id,
        )
    )

    await rate_limit.enforce_rate_limit(
        scope="conversation_delete",
        identifier=identifier,
        limit=20,
        window_seconds=60,
    )

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