from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
)
from fastapi.responses import JSONResponse

from api.dependencies.auth import (
    get_current_user,
)

from core.exceptions import (
    DocumentProcessingError,
)

from services.document_service import (
    process_uploaded_document,
)


router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    current_user=Depends(
        get_current_user
    ),
):
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