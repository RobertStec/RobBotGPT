from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse

from api.dependencies.auth import (
    get_current_user,
)

from core.exceptions import (
    DocumentProcessingError,
)

import core.rate_limit as rate_limit

from services.document_service import (
    process_uploaded_document,
)


router = APIRouter()


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    current_user=Depends(
        get_current_user
    ),
):
    user_id = current_user.id

    identifier = (
        rate_limit
        .build_rate_limit_identifier(
            request=request,
            user_id=user_id,
        )
    )

    await rate_limit.enforce_rate_limit(
        scope="document_upload",
        identifier=identifier,
        limit=10,
        window_seconds=600,
    )

    filename = (
        file.filename
        or "uploaded_file"
    )

    try:
        content = await file.read()

    except Exception as exc:
        raise DocumentProcessingError(
            internal_message=str(exc)
        ) from exc

    result = process_uploaded_document(
        filename=filename,
        content=content,
        thread_id=thread_id,
        user_id=current_user.id,
    )

    return JSONResponse({
        "success": True,
        **result,
    })