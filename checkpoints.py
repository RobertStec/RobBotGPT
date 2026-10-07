import sqlite3

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import settings


@dataclass
class CheckpointResource:
    connection: sqlite3.Connection
    checkpointer: SqliteSaver


_CHECKPOINT_RESOURCE_CACHE: dict[
    str,
    CheckpointResource,
] = {}

_CHECKPOINT_CACHE_LOCK = Lock()


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None = None,
) -> Path:
    """
    Resolve the checkpoint database path and ensure
    that its parent directory exists.
    """

    path = Path(
        checkpoint_path
        or settings.checkpoint_db_path
    ).expanduser().resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def create_checkpoint_resource(
    checkpoint_path: str | Path | None = None,
) -> CheckpointResource:
    """
    Create a SQLite connection and LangGraph
    SqliteSaver for the selected checkpoint database.
    """

    path = _resolve_checkpoint_path(
        checkpoint_path
    )

    connection = sqlite3.connect(
        str(path),
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(
        connection
    )

    return CheckpointResource(
        connection=connection,
        checkpointer=checkpointer,
    )


def get_checkpoint_resource(
    checkpoint_path: str | Path | None = None,
) -> CheckpointResource:
    """
    Return a cached checkpoint resource.

    The resource is initialized lazily on first use.
    """

    path = _resolve_checkpoint_path(
        checkpoint_path
    )

    cache_key = str(path)

    with _CHECKPOINT_CACHE_LOCK:
        if (
            cache_key
            not in _CHECKPOINT_RESOURCE_CACHE
        ):
            _CHECKPOINT_RESOURCE_CACHE[
                cache_key
            ] = create_checkpoint_resource(
                path
            )

        return _CHECKPOINT_RESOURCE_CACHE[
            cache_key
        ]


def get_checkpointer(
    checkpoint_path: str | Path | None = None,
) -> SqliteSaver:
    """
    Return the cached LangGraph checkpointer.
    """

    return get_checkpoint_resource(
        checkpoint_path
    ).checkpointer


def delete_thread_checkpoints(
    thread_id: str,
) -> None:
    """
    Delete all LangGraph checkpoints
    associated with a thread.
    """

    checkpointer = get_checkpointer()

    checkpointer.delete_thread(
        thread_id
    )


def clear_checkpoint_resource_cache() -> None:
    """
    Close checkpoint database connections
    and clear cached resources.

    Primarily useful for tests and controlled
    application shutdown.
    """

    with _CHECKPOINT_CACHE_LOCK:
        resources = list(
            _CHECKPOINT_RESOURCE_CACHE.values()
        )

        _CHECKPOINT_RESOURCE_CACHE.clear()

    for resource in resources:
        resource.connection.close()