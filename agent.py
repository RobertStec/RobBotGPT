from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from langgraph.prebuilt import ToolNode, tools_condition

from core.config import (ALLOWED_MODELS, settings,)
from functools import partial

from state import AgentState
from tools import tools
from nodes import (
    chatbot_node,
    router_node,
    route_after_router,
    rewrite_query_node,
    rerank_node,
    no_answer_node,
    route_after_rerank,
    retrieve_node,
    generate_rag_node,
)

from checkpoints import (
    get_checkpointer,
    delete_thread_checkpoints,
)




DEFAULT_MODEL = settings.openai_model


SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named RobBotGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Answer questions about uploaded documents through the RAG workflow.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about current weather, use get_current_weather.
- If the user asks for the current/latest stock price, use get_stock_price.
- If the user asks to buy or purchase stock, use purchase_stock.
- If the user asks about latest news, current events, recent updates,
  current people, current versions, new releases, or other time-sensitive
  information not handled by a dedicated tool, use Tavily Search.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- Be clear, helpful, and concise.
"""



def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name




def build_agent(model_name: str):
    """
    Build one LangGraph agent for a selected OpenAI model.
    """

    selected_model = normalize_model_name(model_name)

    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        model=selected_model,
        temperature=0.3,
        streaming=True,
        api_key=settings.openai_api_key,
    )

    llm_with_tools = llm.bind_tools(tools)

    chatbot = partial(
        chatbot_node,
        llm_with_tools=llm_with_tools,
        system_prompt=SYSTEM_PROMPT
    )

    rewrite_query = partial(
        rewrite_query_node,
        llm=llm,
    )

    rerank = partial(
        rerank_node,
        llm=llm,
    )

    generate_rag = partial(
        generate_rag_node,
        llm=llm
    )


    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    # =========================
    # Nodes
    # =========================

    workflow.add_node("router", router_node)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank)
    workflow.add_node("generate", generate_rag)
    workflow.add_node("no_answer", no_answer_node)
    workflow.add_node("tools", tool_node)

    # =========================
    # Entry point
    # =========================

    workflow.add_edge(START, "router")

    # =========================
    # Router
    # =========================

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "chatbot": "chatbot",
            "document": "rewrite_query",
        }
    )

    # =========================
    # RAG workflow
    # =========================

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "rerank")

    workflow.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {
            "generate": "generate",
            "no_answer": "no_answer",
        },
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("no_answer", END)

    # =========================
    # Normal agent workflow
    # =========================

    workflow.add_conditional_edges("chatbot", tools_condition)

    workflow.add_edge("tools", "chatbot")

    checkpointer = get_checkpointer()

    return workflow.compile(checkpointer=checkpointer)


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]



def clear_agent_cache() -> None:
    """
    Clear cached compiled LangGraph agents.

    Used during application shutdown and tests.
    """

    _AGENT_CACHE.clear()