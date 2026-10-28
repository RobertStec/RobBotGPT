from dataclasses import dataclass
from threading import Lock

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from sqlalchemy.engine import make_url

from langgraph.checkpoint.postgres import (
    PostgresSaver,
)

from core.config import settings




@dataclass
class CheckpointResource:
    pool: ConnectionPool
    checkpointer: PostgresSaver


_CHECKPOINT_RESOURCE_CACHE: dict[
    str,
    CheckpointResource,
] = {}

_CHECKPOINT_CACHE_LOCK = Lock()



def build_checkpoint_thread_id(
    thread_id: str,
    *,
    user_id: int,
) -> str:
    """
    Build an internal LangGraph checkpoint thread ID
    isolated by authenticated user.
    """

    if not thread_id:
        raise ValueError(
            "thread_id is required."
        )

    if (
        not isinstance(user_id, int)
        or isinstance(user_id, bool)
        or user_id < 1
    ):
        raise ValueError(
            "user_id must be a positive integer."
        )

    return (
        f"user:{user_id}:"
        f"thread:{thread_id}"
    )




def _resolve_checkpoint_database_url(
    database_url: str | None = None,
) -> str:
    """
    Resolve the PostgreSQL connection URL used
    by the LangGraph checkpoint store.

    SQLAlchemy uses:
        postgresql+psycopg://...

    Psycopg expects:
        postgresql://...
    """

    url = make_url(
        database_url
        or settings.database_url
    )

    if (
        url.get_backend_name()
        != "postgresql"
    ):
        raise ValueError(
            "LangGraph PostgreSQL checkpointing "
            "requires a PostgreSQL database URL."
        )

    return (
        url.set(
            drivername="postgresql"
        )
        .render_as_string(
            hide_password=False
        )
    )




def create_checkpoint_resource(
    database_url: str | None = None,
) -> CheckpointResource:
    """
    Create PostgreSQL connection pool and
    LangGraph PostgresSaver.
    """

    conninfo = (
        _resolve_checkpoint_database_url(
            database_url
        )
    )

    pool = ConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=5,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=True,
    )

    try:
        checkpointer = PostgresSaver(
            pool
        )

        checkpointer.setup()

    except Exception:
        pool.close()
        raise

    return CheckpointResource(
        pool=pool,
        checkpointer=checkpointer,
    )




def get_checkpoint_resource(
    database_url: str | None = None,
) -> CheckpointResource:
    """
    Return cached checkpoint resource.

    The resource is initialized lazily
    on first use.
    """

    cache_key = (
        _resolve_checkpoint_database_url(
            database_url
        )
    )

    with _CHECKPOINT_CACHE_LOCK:
        if (
            cache_key
            not in _CHECKPOINT_RESOURCE_CACHE
        ):
            _CHECKPOINT_RESOURCE_CACHE[
                cache_key
            ] = create_checkpoint_resource(
                cache_key
            )

        return _CHECKPOINT_RESOURCE_CACHE[
            cache_key
        ]




def get_checkpointer(
    database_url: str | None = None,
) -> PostgresSaver:
    """
    Return cached LangGraph checkpointer.
    """

    return get_checkpoint_resource(
        database_url
    ).checkpointer




def delete_thread_checkpoints(
    thread_id: str,
    *,
    user_id: int,
) -> None:
    """
    Delete all LangGraph checkpoints
    associated with one user's conversation.
    """

    checkpoint_thread_id = (
        build_checkpoint_thread_id(
            thread_id,
            user_id=user_id,
        )
    )

    checkpointer = get_checkpointer()

    checkpointer.delete_thread(
        checkpoint_thread_id
    )




def clear_checkpoint_resource_cache() -> None:
    """
    Close PostgreSQL connection pools and
    clear cached checkpoint resources.
    """

    with _CHECKPOINT_CACHE_LOCK:
        resources = list(
            _CHECKPOINT_RESOURCE_CACHE.values()
        )

        _CHECKPOINT_RESOURCE_CACHE.clear()

    for resource in resources:
        resource.pool.close()