from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from state import AgentState

from tools import tools



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

    print(f"[ROUTER] Selected route: {route}")

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



def document_request_node(
    state: AgentState,
    llm,
    system_prompt: str,
):
    """
    Force document search for requests routed
    to the document branch.
    """

    messages = [
        SystemMessage(content=system_prompt)
    ] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }




def chatbot_node(
    state: AgentState,
    llm,
    llm_with_tools,
    system_prompt: str,
):
    """
    Main chatbot node.

    Handles:
    - normal conversation,
    - tool usage,
    - final answer after document search
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
    # Document search already returned its result.
    # Generate the final answer WITHOUT tools.
    # -------------------------------------------------

    if (
        isinstance(last_message, ToolMessage)
        and last_message.name == "search_uploaded_documents"
    ):
        response = llm.invoke(messages)

    # -------------------------------------------------
    # Normal chatbot / tool behavior.
    # -------------------------------------------------

    else:
        response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }