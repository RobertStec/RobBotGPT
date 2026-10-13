from core.config import settings

import logging

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from state import AgentState

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from rag import (
    build_rag_context,
    retrieve_candidates_from_rag,
)

from reranker import (
    assess_query_answerability,
    filter_relevant_candidates,
    rerank_candidates,
)



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




def rewrite_query_node(
    state: AgentState,
    llm,
):
    """
    Rewrite the latest document question into
    a standalone semantic search query.

    The rewritten query is used only for retrieval.
    It does not replace the user's original message.
    """

    original_query = (
        get_latest_user_message(state)
    )

    if not original_query:
        return {
            "rag_query": ""
        }

    history_lines = []

    # Use only recent conversational context.
    for message in state["messages"][-6:]:

        if isinstance(
            message,
            HumanMessage,
        ):
            role = "User"

        elif isinstance(
            message,
            AIMessage,
        ):
            role = "Assistant"

        else:
            continue

        if not isinstance(
            message.content,
            str,
        ):
            continue

        history_lines.append(
            f"{role}: {message.content}"
        )

    conversation_history = (
        "\n".join(history_lines)
    )

    prompt = f"""
Rewrite the user's latest question into a standalone
search query for semantic retrieval from uploaded
documents.

Rules:
- Preserve the original meaning.
- Resolve references using the conversation history.
- Do not answer the question.
- Do not add facts that are not present.
- Keep the query concise.
- Use the same language as the user's latest question.
- Return only the rewritten query.

Conversation:
{conversation_history}

Latest question:
{original_query}
"""

    response = llm.invoke([
        SystemMessage(
            content=(
                "You rewrite document questions "
                "for semantic search."
            )
        ),
        HumanMessage(
            content=prompt
        ),
    ])

    rewritten_query = (
        response.content.strip()
        if isinstance(
            response.content,
            str,
        )
        else ""
    )

    return {
        "rag_query": (
            rewritten_query
            or original_query
        )
    }




def retrieve_node(
    state: AgentState,
    config: RunnableConfig,
    writer: StreamWriter,
):
    """
    Retrieve relevant document context from PostgreSQL with pgvector.
    """

    query = (
        state.get("rag_query")
        or get_latest_user_message(
            state
        )
    )

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

    candidates = retrieve_candidates_from_rag(
        query=query,
        thread_id=thread_id,
        k=settings.rag_retrieval_top_k,
    )

    logger.debug(
        (
            "RAG retrieval completed "
            "thread_id=%s context_chars=%d sources=%d"
        ),
        thread_id,
        len(candidates),
    )

    # -----------------------------------------
    # Custom streaming event: RAG completed
    # -----------------------------------------

    writer({
            "type": "rag_search_end",
            "message": "Document Search completed"
        })

    return {
        "rag_candidates": candidates
    }





def rerank_node(
    state: AgentState,
    llm,
):
    """
    Rerank retrieved RAG candidates and build
    the final document context.
    """

    query = (
        state.get("rag_query")
        or get_latest_user_message(state)
    )

    candidates = state.get(
        "rag_candidates",
        [],
    )

    reranked_candidates = (
        rerank_candidates(
            query=query,
            candidates=candidates,
            llm=llm,
            top_k=settings.rag_final_top_k,
        )
    )

    relevant_candidates = (
        filter_relevant_candidates(
            candidates=reranked_candidates,
            threshold=(
                settings.rag_relevance_threshold
            ),
        )
    )

    if (
        relevant_candidates
        and not assess_query_answerability(
            query=query,
            candidates=relevant_candidates,
            llm=llm,
        )
    ):
        relevant_candidates = []

    result = build_rag_context(
        relevant_candidates,
        max_chars=(
            settings.rag_context_max_chars
        ),
    )

    logger.debug(
        (
            "RAG reranking completed "
            "reranked=%d relevant=%d "
            "threshold=%.2f"
        ),
        len(reranked_candidates),
        len(relevant_candidates),
        settings.rag_relevance_threshold,
    )

    return {
        "rag_context": result["context"],
        "rag_sources": result["sources"],
    }





def route_after_rerank(
    state: AgentState,
) -> str:
    """
    Route to answer generation only when
    relevant RAG context is available.
    """

    context = state.get(
        "rag_context",
        "",
    )

    if (
        isinstance(context, str)
        and context.strip()
    ):
        return "generate"

    return "no_answer"





def no_answer_node(
    state: AgentState,
) -> dict:
    """
    Return a safe answer when no sufficiently
    relevant document context was found.
    """

    language = state.get(
        "speech_language",
        "pl-PL",
    )

    if language == "en-US":
        content = (
            "I couldn't find enough relevant "
            "information in the uploaded documents "
            "to answer this question."
        )
    else:
        content = (
            "Nie znalazłem wystarczająco istotnych "
            "informacji w przesłanych dokumentach, "
            "aby odpowiedzieć na to pytanie."
        )

    return {
        "messages": [
            AIMessage(
                content=content
            )
        ]
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

Answer the user's question using only the retrieved
document context below.

Strict grounding rules:
- Use only facts explicitly supported by the retrieved
  document context.
- Do not infer, assume, guess, or derive missing facts
  from related information.
- Do not treat dates of phases, pilots, meetings, or
  milestones as dates of the whole project unless the
  context explicitly says so.
- A project deadline does not imply when the project
  started.
- A planned deadline does not prove whether a project
  is delayed.
- Information about one named entity must not be used
  as information about a different named entity.
- If the question contains multiple parts and the
  context answers only some of them, answer the
  supported parts and explicitly state which requested
  information is not provided in the document.
- If the context does not contain enough information
  to answer any substantive part of the question,
  clearly say that the information is not available
  in the document.
- Do not fill missing information using general
  knowledge.
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