from core.environment import configure_environment

configure_environment()

from pathlib import Path
import uvicorn

from fastapi import (
    FastAPI,
    Request,
    HTTPException,
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.routers.chat import router as chat_router
from api.routers.conversations import router as conversations_router
from api.routers.documents import router as documents_router
from api.routers.auth import router as auth_router

from core.config import settings
from core.lifecycle import lifespan

from core.exceptions import (
    AppError,
    app_error_handler,
)

from sqlalchemy import text

from database import get_engine





BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"

TEMPLATES_DIR = BASE_DIR / "templates"


app = FastAPI(lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(
        directory=str(
            STATIC_DIR
        )
    ),
    name="static",
)

app.add_exception_handler(
    AppError,
    app_error_handler,
)

app.include_router(
    auth_router
)

app.include_router(
    conversations_router
)

app.include_router(
    documents_router
)

app.include_router(
    chat_router
)

templates = Jinja2Templates(
    directory=str(
        TEMPLATES_DIR
    )
)




@app.get("/health")
async def health():
    """
    Application healthcheck.

    Verifies that the FastAPI application
    can communicate with the database.
    """

    try:
        engine = get_engine()

        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "unavailable",
            },
        ) from exc
    return {
        "status": "healthy",
        "database": "connected",
    }




@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )



if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )