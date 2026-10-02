from typing import Annotated, Literal, Any
from typing_extensions import TypedDict, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state of the RobBotGPT LangGraph.
    """

    # Conversation history.
    # add_messages makes LangGraph append new messages
    # instead of replacing the whole list.
    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]

    # Model selected by the user.
    selected_model: NotRequired[str]

    # Language selected for speech recognition.
    speech_language: NotRequired[
        Literal["pl-PL", "en-US"]
    ]

    # Route selected by the router node.
    route: NotRequired[
        Literal["chatbot", "document"]
    ]

    # Context retrieved from the vector database.
    rag_context: NotRequired[str]

    # Sources used by the RAG workflow.
    rag_sources: NotRequired[
        list[dict[str, Any]]
]