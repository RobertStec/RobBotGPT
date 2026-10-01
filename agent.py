import os
import sqlite3
from pathlib import Path
from functools import partial

from dotenv import load_dotenv
import certifi

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START
from state import AgentState

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools
from nodes import chatbot_node

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


Path("data").mkdir(exist_ok=True)
CHECKPOINT_DB_PATH = "data/langgraph_checkpoints.sqlite"


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ALLOWED_MODELS = {
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
}


SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named RobBotGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
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
- If the user asks about an uploaded document, use search_uploaded_documents.
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


# Funkcja chatbot_node jest zagnieżdona tymczasowo w funkcji build_agent

def build_agent(model_name: str):
    """
    Build one LangGraph agent for a selected OpenAI model.
    """

    selected_model = normalize_model_name(model_name)

    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        model=selected_model,
        temperature=0.3,
        streaming=True
    )

    llm_with_tools = llm.bind_tools(tools)

    chatbot = partial(
        chatbot_node,
        llm=llm,
        llm_with_tools=llm_with_tools,
        system_prompt=SYSTEM_PROMPT
    )

    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        CHECKPOINT_DB_PATH,
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

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



def delete_thread_checkpoints(thread_id: str) -> None:
    """
    Delete all LangGraph checkpoints associated with a thread.
    """

    conn = sqlite3.connect(
        CHECKPOINT_DB_PATH,
        check_same_thread=False
    )

    try:
        checkpointer = SqliteSaver(conn)

        checkpointer.delete_thread(thread_id)

    finally:
        conn.close()
