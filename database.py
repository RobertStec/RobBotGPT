from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    inspect,
    text,
)

from sqlalchemy.orm import declarative_base, sessionmaker

import json




Path("data").mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///data/chatbot_memory.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


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




def init_db():
    Base.metadata.create_all(
        bind=engine
    )

    inspector = inspect(engine)

    columns = {
        column["name"]
        for column
        in inspector.get_columns(
            "chat_messages"
        )
    }

    if "sources" not in columns:

        with engine.begin() as connection:

            connection.execute(
                text(
                    "ALTER TABLE chat_messages "
                    "ADD COLUMN sources TEXT"
                )
            )


def create_or_update_conversation(thread_id: str, first_message: str | None = None):
    db = SessionLocal()

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
    db = SessionLocal()

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
    db = SessionLocal()

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
    db = SessionLocal()

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
    db = SessionLocal()

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
    db = SessionLocal()

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



def delete_conversation(thread_id: str) -> bool:
    db = SessionLocal()

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