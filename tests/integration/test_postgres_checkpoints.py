import os
import uuid

import pytest
import checkpoints

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

from langgraph.types import Command

from state import AgentState
from tools import purchase_stock






def hitl_chatbot_node(
    state: AgentState,
):
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



@pytest.fixture
def postgres_checkpoint_database(
    monkeypatch,
):
    database_url = os.getenv(
        "TEST_DATABASE_URL"
    )

    if not database_url:
        pytest.skip(
            "TEST_DATABASE_URL is not configured."
        )

    checkpoints.clear_checkpoint_resource_cache()

    monkeypatch.setattr(
        checkpoints.settings,
        "database_url",
        database_url,
    )

    yield database_url

    checkpoints.clear_checkpoint_resource_cache()




def build_hitl_graph(
    checkpointer,
):
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




def test_postgres_checkpoint_survives_resource_restart(
    postgres_checkpoint_database,
):
    user_id = 101

    thread_id = (
        "postgres-hitl-"
        f"{uuid.uuid4()}"
    )

    checkpoint_thread_id = (
        checkpoints.build_checkpoint_thread_id(
            thread_id,
            user_id=user_id,
        )
    )

    config = {
        "configurable": {
            "thread_id": checkpoint_thread_id,
            "conversation_thread_id": thread_id,
            "user_id": user_id,
        }
    }

    checkpoints.clear_checkpoint_resource_cache()

    try:
        # =====================================
        # FIRST PROCESS / RESOURCE
        # =====================================

        first_checkpointer = (
            checkpoints.get_checkpointer()
        )

        first_graph = build_hitl_graph(
            first_checkpointer
        )

        result = first_graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Buy 5 shares of AAPL"
                        )
                    )
                ]
            },
            config=config,
        )

        assert "__interrupt__" in result

        interrupts = (
            result["__interrupt__"]
        )

        assert len(interrupts) == 1

        payload = interrupts[0].value

        assert (
            payload["type"]
            == "stock_purchase_approval"
        )

        assert payload["symbol"] == "AAPL"
        assert payload["quantity"] == 5

        # =====================================
        # SIMULATE APPLICATION RESTART
        # =====================================

        checkpoints.clear_checkpoint_resource_cache()

        # =====================================
        # SECOND PROCESS / RESOURCE
        # =====================================

        second_checkpointer = (
            checkpoints.get_checkpointer()
        )

        second_graph = build_hitl_graph(
            second_checkpointer
        )

        resumed = second_graph.invoke(
            Command(
                resume=True
            ),
            config=config,
        )

        messages = resumed[
            "messages"
        ]

        final_message = messages[-1]

        assert isinstance(
            final_message,
            AIMessage,
        )

        assert (
            "success"
            in final_message.content.lower()
        )

    finally:
        checkpoints.delete_thread_checkpoints(
            thread_id,
            user_id=user_id,
        )

        checkpoints.clear_checkpoint_resource_cache()




def test_postgres_checkpoint_rows_are_deleted_with_thread(
    postgres_checkpoint_database,
):
    user_id = 202

    thread_id = (
        "postgres-delete-"
        f"{uuid.uuid4()}"
    )

    checkpoint_thread_id = (
        checkpoints.build_checkpoint_thread_id(
            thread_id,
            user_id=user_id,
        )
    )

    config = {
        "configurable": {
            "thread_id": checkpoint_thread_id,
            "conversation_thread_id": thread_id,
            "user_id": user_id,
        }
    }

    checkpoints.clear_checkpoint_resource_cache()

    try:
        resource = (
            checkpoints.get_checkpoint_resource()
        )

        graph = build_hitl_graph(
            resource.checkpointer
        )

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Buy 5 shares of AAPL"
                        )
                    )
                ]
            },
            config=config,
        )

        assert "__interrupt__" in result

        # =====================================
        # Check physical PostgreSQL rows
        # =====================================

        with resource.pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM checkpoints
                    WHERE thread_id = %s
                    """,
                    (checkpoint_thread_id,),
                )

                checkpoint_count = (
                    cursor.fetchone()["count"]
                )

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM checkpoint_blobs
                    WHERE thread_id = %s
                    """,
                    (checkpoint_thread_id,),
                )

                blob_count = (
                    cursor.fetchone()["count"]
                )

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM checkpoint_writes
                    WHERE thread_id = %s
                    """,
                    (checkpoint_thread_id,),
                )

                write_count = (
                    cursor.fetchone()["count"]
                )

        assert checkpoint_count > 0

        assert (
            checkpoint_count
            + blob_count
            + write_count
            > 0
        )

        # =====================================
        # Delete whole LangGraph thread
        # =====================================

        checkpoints.delete_thread_checkpoints(
            thread_id,
            user_id=user_id,
        )

        # =====================================
        # Verify PostgreSQL cleanup
        # =====================================

        with resource.pool.connection() as conn:
            with conn.cursor() as cursor:
                for table_name in (
                    "checkpoints",
                    "checkpoint_blobs",
                    "checkpoint_writes",
                ):
                    cursor.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM {table_name}
                        WHERE thread_id = %s
                        """,
                        (checkpoint_thread_id,),
                    )

                    count = (
                        cursor.fetchone()["count"]
                    )

                    assert count == 0

    finally:
        # Safe even when rows are already gone.
        try:
            checkpoints.delete_thread_checkpoints(
                thread_id,
                user_id=user_id,
            )
        finally:
            checkpoints.clear_checkpoint_resource_cache()




def test_postgres_checkpoints_are_isolated_by_user_thread_id(
    postgres_checkpoint_database,
):
    conversation_thread_id = (
        "postgres-isolation-"
        f"{uuid.uuid4()}"
    )

    user_a_id = 301
    user_b_id = 302

    checkpoint_thread_a = (
        checkpoints.build_checkpoint_thread_id(
            conversation_thread_id,
            user_id=user_a_id,
        )
    )

    checkpoint_thread_b = (
        checkpoints.build_checkpoint_thread_id(
            conversation_thread_id,
            user_id=user_b_id,
        )
    )

    assert (
        checkpoint_thread_a
        != checkpoint_thread_b
    )

    config_a = {
        "configurable": {
            "thread_id": checkpoint_thread_a,
            "conversation_thread_id": (
                conversation_thread_id
            ),
            "user_id": user_a_id,
        }
    }

    config_b = {
        "configurable": {
            "thread_id": checkpoint_thread_b,
            "conversation_thread_id": (
                conversation_thread_id
            ),
            "user_id": user_b_id,
        }
    }

    checkpoints.clear_checkpoint_resource_cache()

    try:
        resource = (
            checkpoints.get_checkpoint_resource()
        )

        graph = build_hitl_graph(
            resource.checkpointer
        )

        result_a = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Buy 5 shares of AAPL"
                        )
                    )
                ]
            },
            config=config_a,
        )

        assert "__interrupt__" in result_a

        result_b = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Buy 5 shares of AAPL"
                        )
                    )
                ]
            },
            config=config_b,
        )

        assert "__interrupt__" in result_b

        with resource.pool.connection() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM checkpoints
                    WHERE thread_id = %s
                    """,
                    (checkpoint_thread_a,),
                )

                count_a = (
                    cursor.fetchone()["count"]
                )

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM checkpoints
                    WHERE thread_id = %s
                    """,
                    (checkpoint_thread_b,),
                )

                count_b = (
                    cursor.fetchone()["count"]
                )

        assert count_a > 0
        assert count_b > 0

        checkpoints.delete_thread_checkpoints(
            conversation_thread_id,
            user_id=user_a_id,
        )

        with resource.pool.connection() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM checkpoints
                    WHERE thread_id = %s
                    """,
                    (checkpoint_thread_a,),
                )

                remaining_a = (
                    cursor.fetchone()["count"]
                )

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM checkpoints
                    WHERE thread_id = %s
                    """,
                    (checkpoint_thread_b,),
                )

                remaining_b = (
                    cursor.fetchone()["count"]
                )

        assert remaining_a == 0
        assert remaining_b > 0

    finally:
        try:
            checkpoints.delete_thread_checkpoints(
                conversation_thread_id,
                user_id=user_a_id,
            )

            checkpoints.delete_thread_checkpoints(
                conversation_thread_id,
                user_id=user_b_id,
            )

        finally:
            checkpoints.clear_checkpoint_resource_cache()