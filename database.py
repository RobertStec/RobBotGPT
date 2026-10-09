from core.config import settings

from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    text,
)

from sqlalchemy.engine import (
    Engine,
    make_url,
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
)

from pgvector.sqlalchemy import Vector

import json




Base = declarative_base()

_ENGINE_CACHE: dict[str, Engine] = {}
_SESSION_FACTORY_CACHE: dict[str, sessionmaker] = {}



def _ensure_sqlite_directory(
    database_url: str,
) -> None:
    """
    Create the parent directory for a file-based
    SQLite database when necessary.
    """

    url = make_url(database_url)

    if url.get_backend_name() != "sqlite":
        return

    database_path = url.database

    if (
        not database_path
        or database_path == ":memory:"
    ):
        return

    from pathlib import Path

    Path(database_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_database_engine(
    database_url: str,
) -> Engine:
    """
    Create an SQLAlchemy engine.

    PostgreSQL is the runtime database backend.
    SQLite support is retained for isolated tests.
    """

    _ensure_sqlite_directory(
        database_url
    )

    url = make_url(
        database_url
    )

    engine_kwargs = {
        "pool_pre_ping": True,
    }

    if (
        url.get_backend_name()
        == "sqlite"
    ):
        engine_kwargs[
            "connect_args"
        ] = {
            "check_same_thread": False
        }

    return create_engine(
        database_url,
        **engine_kwargs,
    )


def get_engine(
    database_url: str | None = None,
) -> Engine:
    """
    Return a cached SQLAlchemy engine.

    The engine is initialized lazily on first use.
    """

    resolved_url = (
        database_url
        or settings.database_url
    )

    if (
        resolved_url
        not in _ENGINE_CACHE
    ):
        _ENGINE_CACHE[
            resolved_url
        ] = create_database_engine(
            resolved_url
        )

    return _ENGINE_CACHE[
        resolved_url
    ]


def get_session_factory(
    database_url: str | None = None,
):
    """
    Return a cached SQLAlchemy session factory.
    """

    resolved_url = (
        database_url
        or settings.database_url
    )

    if (
        resolved_url
        not in _SESSION_FACTORY_CACHE
    ):
        _SESSION_FACTORY_CACHE[
            resolved_url
        ] = sessionmaker(
            bind=get_engine(
                resolved_url
            ),
            autoflush=False,
            autocommit=False,
        )

    return _SESSION_FACTORY_CACHE[
        resolved_url
    ]


def create_session():
    """
    Create a new database session using
    the configured session factory.
    """

    return get_session_factory()()


def clear_database_resource_cache() -> None:
    """
    Dispose cached database engines and clear
    session factories.

    Primarily useful in tests and controlled
    reconfiguration.
    """

    _SESSION_FACTORY_CACHE.clear()

    for engine in _ENGINE_CACHE.values():
        engine.dispose()

    _ENGINE_CACHE.clear()



EMBEDDING_DIMENSION = 1536


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    sources = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    memory = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class RagDocument(Base):
    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True,)
    thread_id = Column(String, nullable=False, index=True)
    source_name = Column(String, nullable=False)
    stored_name = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer,
        ForeignKey("rag_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(
        Vector(EMBEDDING_DIMENSION), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name=(
                "uq_rag_chunks_"
                "document_chunk"
            ),
        ),
        Index(
            "ix_rag_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={
                "embedding":
                    "vector_cosine_ops"
            },
        ),
    )



def init_db() -> None:
    """
    Verify that the configured database is reachable.

    Database schema creation and migrations are
    managed exclusively by Alembic.
    """

    engine = get_engine()

    with engine.connect() as connection:
        connection.execute(
            text("SELECT 1")
        )



def create_or_update_conversation(thread_id: str, first_message: str | None = None):
    db = create_session()

    try:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if not conversation:
            title = "New Chat"

            if first_message:
                title = first_message.strip()[:40]
                if len(first_message.strip()) > 40:
                    title += "..."

            conversation = Conversation(
                thread_id=thread_id,
                title=title,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            db.add(conversation)

        else:
            conversation.updated_at = datetime.utcnow()

        db.commit()

    finally:
        db.close()


def list_conversations():
    db = create_session()

    try:
        return (
            db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    finally:
        db.close()


def save_chat_message(
    thread_id: str,
    role: str,
    content: str,
    sources: list | None = None,
):
    db = create_session()
    try:

        sources_json = (
            json.dumps(
                sources,
                ensure_ascii=False
            )
            if sources
            else None
        )

        msg = ChatMessage(
            thread_id=thread_id,
            role=role,
            content=content,
            sources=sources_json,
            created_at=datetime.utcnow()
        )

        db.add(msg)

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.thread_id
                == thread_id
            )
            .first()
        )

        if conversation:
            conversation.updated_at = (
                datetime.utcnow()
            )

        db.commit()

    finally:
        db.close()

        


def get_chat_history(thread_id: str):
    db = create_session()

    try:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

    finally:
        db.close()


def save_memory(thread_id: str, memory: str):
    db = create_session()

    try:
        item = LongTermMemory(
            thread_id=thread_id,
            memory=memory,
            created_at=datetime.utcnow()
        )

        db.add(item)
        db.commit()

        return "Memory saved successfully."

    finally:
        db.close()


def search_memory(thread_id: str, query: str):
    db = create_session()

    try:
        memories = (
            db.query(LongTermMemory)
            .filter(LongTermMemory.thread_id == thread_id)
            .order_by(LongTermMemory.created_at.desc())
            .limit(20)
            .all()
        )

        if not memories:
            return "No saved memory found."

        return "\n".join([f"- {m.memory}" for m in memories])

    finally:
        db.close()




def conversation_exists(
    thread_id: str,
) -> bool:
    """
    Check whether a conversation exists
    without modifying any data.
    """

    db = create_session()

    try:
        return (
            db.query(Conversation.id)
            .filter(
                Conversation.thread_id
                == thread_id
            )
            .first()
            is not None
        )

    finally:
        db.close()





def delete_conversation(thread_id: str) -> bool:
    db = create_session()

    try:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if not conversation:
            return False

        # Delete chat messages
        db.query(ChatMessage).filter(
            ChatMessage.thread_id == thread_id
        ).delete(synchronize_session=False)

        # Delete long-term memory connected with this thread
        db.query(LongTermMemory).filter(
            LongTermMemory.thread_id == thread_id
        ).delete(synchronize_session=False)

        # Delete conversation
        db.delete(conversation)

        db.commit()

        return True

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()