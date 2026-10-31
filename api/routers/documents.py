from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Request,
    UploadFile,
)

from core.config import settings

from fastapi.responses import JSONResponse

from api.dependencies.auth import get_current_user

from api.schemas.common import ThreadId

from core.exceptions import (
    AppError,
    DocumentProcessingError,
    EmptyUploadError,
    UploadTooLargeError,
)

import core.rate_limit as rate_limit

from services.document_service import (
    process_uploaded_document,
    validate_upload_filename,
)

from typing import Annotated




router = APIRouter()


UPLOAD_READ_CHUNK_SIZE = 1024 * 1024



async def _read_upload_with_limit(
    file: UploadFile,
    *,
    max_bytes: int,
) -> bytes:
    """
    Read an uploaded file with a hard size limit
    and always release the underlying temporary file.
    """

    content = bytearray()

    try:
        while True:
            remaining = (
                max_bytes
                - len(content)
            )

            chunk = await file.read(
                min(
                    UPLOAD_READ_CHUNK_SIZE,
                    remaining + 1,
                )
            )

            if not chunk:
                break

            content.extend(chunk)

            if len(content) > max_bytes:
                raise UploadTooLargeError(
                    max_bytes=max_bytes
                )

        if not content:
            raise EmptyUploadError()

        return bytes(content)

    finally:
        await file.close()



@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    thread_id: Annotated[
        ThreadId,
        Form(),
    ] = ...,
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

    filename = validate_upload_filename(
        file.filename
        or ""
    )

    try:
        content = await _read_upload_with_limit(
            file,
            max_bytes=settings.upload_max_bytes,
        )

    except AppError:
        raise

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