import importlib
import json

import pytest

import database

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from langgraph.graph import (
    StateGraph,
    START,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from state import AgentState
from tools import purchase_stock



@pytest.fixture
def chat_router_module(
    integration_app,
):
    return importlib.import_module(
        "api.routers.chat"
    )


# =========================================================
# SSE helper
# =========================================================

def parse_sse_events(response):
    events = []

    for block in response.text.split("\n\n"):
        block = block.strip()

        if not block:
            continue

        data_lines = [
            line.removeprefix("data:").strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]

        if not data_lines:
            continue

        payload = "\n".join(
            data_lines
        )

        events.append(
            json.loads(payload)
        )

    return events


# =========================================================
# Deterministic chatbot node
# =========================================================

def hitl_chatbot_node(state: AgentState):
    """
    Deterministic replacement for the LLM.

    First execution:
        HumanMessage
            ->
        purchase_stock tool call

    After HITL resume:
        ToolMessage
            ->
        final AI response
    """

    last_message = (
        state["messages"][-1]
    )

    if isinstance(
        last_message,
        HumanMessage,
    ):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "purchase_stock",
                            "args": {
                                "symbol": "AAPL",
                                "quantity": 5,
                            },
                            "id": "purchase-call-001",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    if isinstance(
        last_message,
        ToolMessage,
    ):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Purchase result: "
                        f"{last_message.content}"
                    )
                )
            ]
        }

    return {
        "messages": []
    }


# =========================================================
# Real LangGraph HITL fixture
# =========================================================

@pytest.fixture
def hitl_agent():
    checkpointer = InMemorySaver()

    workflow = StateGraph(
        AgentState
    )

    workflow.add_node(
        "chatbot",
        hitl_chatbot_node,
    )

    workflow.add_node(
        "tools",
        ToolNode([
            purchase_stock
        ]),
    )

    workflow.add_edge(
        START,
        "chatbot",
    )

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    workflow.add_edge(
        "tools",
        "chatbot",
    )

    return workflow.compile(
        checkpointer=checkpointer
    )




# Test całego cyklu Approve / Decline

@pytest.mark.parametrize(
    (
        "approved",
        "expected_status",
    ),
    [
        (
            True,
            "success",
        ),
        (
            False,
            "cancelled",
        ),
    ],
)
def test_hitl_interrupt_and_resume(
    api_client,
    chat_router_module,
    hitl_agent,
    mocker,
    approved,
    expected_status,
):
    mocker.patch.object(
        chat_router_module,
        "get_agent",
        return_value=hitl_agent,
    )

    thread_id = (
        f"thread-hitl-{approved}"
    )

    # =====================================================
    # STEP 1
    # Initial request
    # =====================================================

    response = api_client.post(
        "/chat/stream",
        json={
            "message": (
                "Buy 5 shares of AAPL"
            ),
            "thread_id": thread_id,
            "model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200

    events = parse_sse_events(
        response
    )

    event_types = [
        event["type"]
        for event in events
    ]

    # LLM requested the tool.
    assert "tool_start" in event_types

    # Tool execution was interrupted.
    assert "interrupt" in event_types

    # Tool did not finish yet.
    assert "tool_end" not in event_types

    # Stream itself finishes.
    assert events[-1] == {
        "type": "done",
        "done": True,
    }

    # =====================================================
    # Interrupt payload
    # =====================================================

    interrupt_event = next(
        event
        for event in events
        if event["type"] == "interrupt"
    )

    payload = (
        interrupt_event["payload"]
    )

    assert (
        payload["type"]
        == "stock_purchase_approval"
    )

    assert (
        payload["action"]
        == "purchase_stock"
    )

    assert (
        payload["symbol"]
        == "AAPL"
    )

    assert (
        payload["quantity"]
        == 5
    )

    assert (
        "approve purchasing"
        in payload["message"].lower()
    )

    # =====================================================
    # No assistant answer before approval
    # =====================================================

    history_before_resume = (
        database.get_chat_history(
            thread_id
        )
    )

    assert len(
        history_before_resume
    ) == 1

    assert (
        history_before_resume[0].role
        == "user"
    )

    # =====================================================
    # STEP 2
    # Resume the SAME graph/thread
    # =====================================================

    resume_response = api_client.post(
        "/chat/stream",
        json={
            "thread_id": thread_id,
            "model": "gpt-4o-mini",
            "resume": approved,
        },
    )

    assert (
        resume_response.status_code
        == 200
    )

    resume_events = parse_sse_events(
        resume_response
    )

    resume_types = [
        event["type"]
        for event in resume_events
    ]

    # Tool can now complete.
    assert "tool_end" in resume_types

    # Agent generates final answer.
    assert "token" in resume_types

    assert resume_events[-1] == {
        "type": "done",
        "done": True,
    }

    # =====================================================
    # Final assistant text
    # =====================================================

    final_text = "".join(
        event["token"]
        for event in resume_events
        if event["type"] == "token"
    )

    assert (
        expected_status
        in final_text.lower()
    )

    # =====================================================
    # Persistence
    # =====================================================

    history_after_resume = (
        database.get_chat_history(
            thread_id
        )
    )

    assert len(
        history_after_resume
    ) == 2

    assert (
        history_after_resume[0].role
        == "user"
    )

    assert (
        history_after_resume[1].role
        == "assistant"
    )

    # Resume must NOT save another user message.
    user_messages = [
        message
        for message
        in history_after_resume
        if message.role == "user"
    ]

    assert len(user_messages) == 1




# Test thread_id -resume musi dotyczyć tego samego wątku

def test_hitl_resume_requires_same_thread_id(
    api_client,
    chat_router_module,
    hitl_agent,
    mocker,
):
    mocker.patch.object(
        chat_router_module,
        "get_agent",
        return_value=hitl_agent,
    )

    # -----------------------------------------
    # Interrupt thread A
    # -----------------------------------------

    first_response = api_client.post(
        "/chat/stream",
        json={
            "message": (
                "Buy 5 shares of AAPL"
            ),
            "thread_id": "thread-A",
            "model": "gpt-4o-mini",
        },
    )

    assert (
        first_response.status_code
        == 200
    )

    first_events = parse_sse_events(
        first_response
    )

    assert any(
        event["type"] == "interrupt"
        for event in first_events
    )

    # -----------------------------------------
    # Incorrectly resume thread B
    # -----------------------------------------

    wrong_resume = api_client.post(
        "/chat/stream",
        json={
            "thread_id": "thread-B",
            "model": "gpt-4o-mini",
            "resume": True,
        },
    )

    assert (
        wrong_resume.status_code
        == 200
    )

    events = parse_sse_events(
        wrong_resume
    )

    assert any(
        event["type"] == "error"
        for event in events
    )

    assert events[-1] == {
        "type": "done",
        "done": True,
    }
