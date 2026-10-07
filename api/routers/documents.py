from fastapi import (
    APIRouter,
    File,
    Form,
    UploadFile,
)
from fastapi.responses import JSONResponse

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
    )

    return JSONResponse({
        "success": True,
        **result,
    })