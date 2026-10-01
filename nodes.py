from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import MessagesState

from tools import tools


def is_document_question(state: MessagesState) -> bool:
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


def chatbot_node(
    state: MessagesState,
    llm,
    llm_with_tools,
    system_prompt: str,
):
    """
    Main chatbot node.

    Handles:
    - normal conversation,
    - tool usage,
    - forced document search,
    - final answer after RAG.
    """

    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    last_message = (
        state["messages"][-1]
        if state["messages"]
        else None
    )

    # -------------------------------------------------
    # RAG has already returned its result.
    # Generate final answer WITHOUT calling tools again.
    # -------------------------------------------------

    if (
        isinstance(last_message, ToolMessage)
        and last_message.name == "search_uploaded_documents"
    ):
        response = llm.invoke(messages)

    # -------------------------------------------------
    # New question about uploaded document.
    # Force document search exactly once.
    # -------------------------------------------------

    elif is_document_question(state):

        llm_for_request = llm.bind_tools(
            tools,
            tool_choice="search_uploaded_documents"
        )

        response = llm_for_request.invoke(messages)

    # -------------------------------------------------
    # Normal agent behavior.
    # -------------------------------------------------

    else:
        response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }