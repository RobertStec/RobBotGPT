import logging

from core.config import settings

from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)
from state import AgentState

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from rag import retrieve_from_rag




logger = logging.getLogger(__name__)



def is_document_question(state: AgentState) -> bool:
    """
    Check whether the current user message refers
    to an uploaded document or file.
    """

    messages = state["messages"]

    if not messages:
        return False

    last_message = messages[-1]

    # Only a user message can trigger forced RAG.
    if not isinstance(last_message, HumanMessage):
        return False

    content = last_message.content

    if not isinstance(content, str):
        return False

    text = content.lower()

    document_keywords = [
        # English
        "document",
        "uploaded document",
        "uploaded file",
        "uploaded pdf",
        "pdf",
        "file",
        "attachment",
        "summarize the document",

        # Polish
        "dokument",
        "wgrany dokument",
        "wgranym dokumencie",
        "wgranego dokumentu",
        "wgrany pdf",
        "wgranym pdf",
        "plik",
        "wgrany plik",
        "załącznik",
        "podsumuj dokument",
    ]

    return any(
        keyword in text
        for keyword in document_keywords
    )




def router_node(state: AgentState) -> dict:
    """
    Decide which branch of the graph should handle
    the current user request.
    """

    if is_document_question(state):
        route = "document"
    else:
        route = "chatbot"

    logger.debug(
        "Router selected route=%s",
        route,
    )

    return {
        "route": route
    }




def route_after_router(state: AgentState) -> str:
    """
    Return the route selected by router_node.
    """

    return state.get(
        "route",
        "chatbot"
    )




def get_latest_user_message(state: AgentState) -> str:
    """
    Return the latest HumanMessage content.
    """

    for message in reversed(state["messages"]):

        if isinstance(message, HumanMessage):

            if isinstance(message.content, str):
                return message.content

    return ""




def retrieve_node(
    state: AgentState,
    config: RunnableConfig,
    writer: StreamWriter,
):
    """
    Retrieve relevant document context from PostgreSQL with pgvector.
    """

    query = get_latest_user_message(state)

    thread_id = (
        config
        .get("configurable", {})
        .get("thread_id")
    )

    if not thread_id:
        raise ValueError(
            "thread_id is missing from LangGraph config."
        )

    # -----------------------------------------
    # Custom streaming event: RAG started
    # -----------------------------------------

    writer({
        "type": "rag_search_start",
        "message": "Searching documents..."
    })

    logger.info(
        "RAG retrieval started thread_id=%s",
        thread_id,
    )

    result = retrieve_from_rag(
        query=query,
        thread_id=thread_id,
        k=settings.rag_top_k,
    )

    context = result["context"]
    sources = result["sources"]

    logger.debug(
        (
            "RAG retrieval completed "
            "thread_id=%s context_chars=%d sources=%d"
        ),
        thread_id,
        len(context),
        len(sources),
    )

    # -----------------------------------------
    # Custom streaming event: RAG completed
    # -----------------------------------------

    writer({
            "type": "rag_search_end",
            "message": "Document Search completed"
        })

    return {
        "rag_context": context,
        "rag_sources": sources
    }




def generate_rag_node(
    state: AgentState,
    llm,
    writer: StreamWriter,
):
    """
    Generate the final answer using
    the retrieved RAG context.
    """

    context = state.get(
        "rag_context",
        ""
    )

    rag_system_prompt = f"""
You are RobBotGPT.

Answer the user's question using the retrieved
document context below.

Rules:
- Base the answer on the retrieved context.
- Do not invent information that is not supported
  by the document context.
- If the context does not contain enough information,
  say so clearly.
- Answer in the same language as the user's question.
- Be clear and concise.

Retrieved document context:

{context}
"""

    messages = [
        SystemMessage(
            content=rag_system_prompt
        )
    ] + state["messages"]

    response = llm.invoke(messages)

    sources = state.get(
        "rag_sources",
        []
    )

    if sources:
        writer({
            "type": "rag_sources",
            "sources": sources
        })


    return {
        "messages": [response]
    }




def chatbot_node(
    state: AgentState,
    llm_with_tools,
    system_prompt: str,
):
    """
    Handle normal conversation
    and normal tool usage.
    """

    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }