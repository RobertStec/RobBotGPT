from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.config import settings

from agent import clear_agent_cache

from checkpoints import (
    clear_checkpoint_resource_cache,
)

from database import (
    init_db,
    clear_database_resource_cache,
)

from rag import (
    clear_rag_resource_cache,
)

import logging
from core.logging import configure_logging



logger = logging.getLogger(__name__)



def initialize_resources() -> None:
    configure_logging()

    logger.info(
        "Initializing application resources"
    )

    settings.data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    init_db()

    logger.info(
        "Application resources initialized"
    )




def shutdown_resources() -> None:
    logger.info(
        "Shutting down application resources"
    )

    clear_agent_cache()

    clear_checkpoint_resource_cache()

    clear_rag_resource_cache()

    clear_database_resource_cache()

    logger.info(
        "Application resources shut down"
    )



@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    FastAPI application lifecycle.
    """

    initialize_resources()

    try:
        yield

    finally:
        shutdown_resources()