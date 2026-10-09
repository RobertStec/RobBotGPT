from functools import partial

import pytest

import agent


@pytest.fixture(autouse=True)
def clear_agent_cache():
    """
    Each test starts with an empty agent cache.
    """
    agent._AGENT_CACHE.clear()

    yield

    agent._AGENT_CACHE.clear()


# =========================================================
# normalize_model_name
# =========================================================

@pytest.mark.parametrize(
    "model_name",
    [
        None,
        "",
        "   ",
        "not-a-real-model",
        "gpt-invalid",
    ],
)
def test_normalize_model_name_returns_default_for_invalid_model(
    model_name,
):
    result = agent.normalize_model_name(
        model_name
    )

    assert result == agent.DEFAULT_MODEL


@pytest.mark.parametrize(
    "model_name",
    sorted(agent.ALLOWED_MODELS),
)
def test_normalize_model_name_accepts_allowed_models(
    model_name,
):
    result = agent.normalize_model_name(
        model_name
    )

    assert result == model_name


def test_normalize_model_name_strips_whitespace():
    result = agent.normalize_model_name(
        "  gpt-4o-mini  "
    )

    assert result == "gpt-4o-mini"


# =========================================================
# build_agent
# =========================================================

def test_build_agent_creates_expected_langgraph(
    mocker,
):
    # -----------------------------------------
    # LLM
    # -----------------------------------------

    mock_chat_openai = mocker.patch(
        "agent.ChatOpenAI"
    )

    mock_llm = mocker.Mock()
    mock_llm_with_tools = mocker.Mock()

    mock_chat_openai.return_value = mock_llm

    mock_llm.bind_tools.return_value = (
        mock_llm_with_tools
    )

    # -----------------------------------------
    # ToolNode
    # -----------------------------------------

    mock_tool_node_class = mocker.patch(
        "agent.ToolNode"
    )

    mock_tool_node = mocker.Mock()

    mock_tool_node_class.return_value = (
        mock_tool_node
    )

    # -----------------------------------------
    # StateGraph
    # -----------------------------------------

    mock_state_graph_class = mocker.patch(
        "agent.StateGraph"
    )

    mock_workflow = mocker.Mock()

    mock_state_graph_class.return_value = (
        mock_workflow
    )

    compiled_agent = mocker.Mock()

    mock_workflow.compile.return_value = (
        compiled_agent
    )

    # -----------------------------------------
    # LangGraph checkpointer
    # -----------------------------------------

    
    mock_checkpointer = mocker.Mock()

    mock_get_checkpointer = mocker.patch(
        "agent.get_checkpointer",
        return_value=mock_checkpointer,
    )

    # -----------------------------------------
    # Build agent
    # -----------------------------------------

    result = agent.build_agent(
        "gpt-4o-mini"
    )

    # -----------------------------------------
    # ChatOpenAI configuration
    # -----------------------------------------

    mock_chat_openai.assert_called_once_with(
        model="gpt-4o-mini",
        temperature=0.3,
        streaming=True,
        api_key=agent.settings.openai_api_key,
    )

    mock_llm.bind_tools.assert_called_once_with(
        agent.tools
    )

    # -----------------------------------------
    # StateGraph
    # -----------------------------------------

    mock_state_graph_class.assert_called_once_with(
        agent.AgentState
    )

    mock_tool_node_class.assert_called_once_with(
        agent.tools
    )

    # -----------------------------------------
    # Nodes
    # -----------------------------------------

    node_names = [
        call.args[0]
        for call
        in mock_workflow.add_node.call_args_list
    ]

    assert node_names == [
        "router",
        "chatbot",
        "retrieve",
        "generate",
        "tools",
    ]

    mock_workflow.add_node.assert_any_call(
        "router",
        agent.router_node,
    )

    mock_workflow.add_node.assert_any_call(
        "tools",
        mock_tool_node,
    )

    # -----------------------------------------
    # Check chatbot partial
    # -----------------------------------------

    chatbot_call = (
        mock_workflow
        .add_node
        .call_args_list[1]
    )

    chatbot_partial = chatbot_call.args[1]

    assert isinstance(
        chatbot_partial,
        partial,
    )

    assert (
        chatbot_partial.func
        is agent.chatbot_node
    )

    assert (
        chatbot_partial.keywords[
            "llm_with_tools"
        ]
        is mock_llm_with_tools
    )

    assert (
        chatbot_partial.keywords[
            "system_prompt"
        ]
        == agent.SYSTEM_PROMPT
    )

    # -----------------------------------------
    # Check RAG generation partial
    # -----------------------------------------

    generate_call = (
        mock_workflow
        .add_node
        .call_args_list[3]
    )

    generate_partial = generate_call.args[1]

    assert isinstance(
        generate_partial,
        partial,
    )

    assert (
        generate_partial.func
        is agent.generate_rag_node
    )

    assert (
        generate_partial.keywords["llm"]
        is mock_llm
    )

    # -----------------------------------------
    # Graph edges
    # -----------------------------------------

    mock_workflow.add_edge.assert_any_call(
        agent.START,
        "router",
    )

    mock_workflow.add_edge.assert_any_call(
        "retrieve",
        "generate",
    )

    mock_workflow.add_edge.assert_any_call(
        "generate",
        agent.END,
    )

    mock_workflow.add_edge.assert_any_call(
        "tools",
        "chatbot",
    )

    # -----------------------------------------
    # Conditional routing
    # -----------------------------------------

    mock_workflow.add_conditional_edges.assert_any_call(
        "router",
        agent.route_after_router,
        {
            "chatbot": "chatbot",
            "document": "retrieve",
        },
    )

    mock_workflow.add_conditional_edges.assert_any_call(
        "chatbot",
        agent.tools_condition,
    )

    # -----------------------------------------
    # Checkpointer
    # -----------------------------------------

    mock_workflow.compile.assert_called_once_with(
        checkpointer=mock_checkpointer
    )

    mock_get_checkpointer.assert_called_once_with()

    assert result is compiled_agent




# Test fallback modelu w build_agent()

def test_build_agent_uses_default_model_for_invalid_model(
    mocker,
):
    mock_chat_openai = mocker.patch(
        "agent.ChatOpenAI"
    )

    mock_llm = mocker.Mock()

    mock_llm.bind_tools.return_value = (
        mocker.Mock()
    )

    mock_chat_openai.return_value = (
        mock_llm
    )

    mocker.patch(
        "agent.ToolNode"
    )

    mock_workflow = mocker.patch(
        "agent.StateGraph"
    ).return_value

    mock_workflow.compile.return_value = (
        mocker.Mock()
    )

    mock_checkpointer = mocker.Mock()

    mocker.patch(
        "agent.get_checkpointer",
        return_value=mock_checkpointer,
    )

    agent.build_agent(
        "invalid-model"
    )

    mock_chat_openai.assert_called_once_with(
        model=agent.DEFAULT_MODEL,
        temperature=0.3,
        streaming=True,
        api_key=agent.settings.openai_api_key,
    )

    mock_workflow.compile.assert_called_once_with(
        checkpointer=mock_checkpointer
    )




# Test cache agentów

def test_get_agent_builds_agent_only_once_for_same_model(
    mocker,
):
    compiled_agent = mocker.Mock()

    mock_build_agent = mocker.patch(
        "agent.build_agent",
        return_value=compiled_agent,
    )

    first = agent.get_agent(
        "gpt-4o-mini"
    )

    second = agent.get_agent(
        "gpt-4o-mini"
    )

    assert first is compiled_agent
    assert second is compiled_agent
    assert first is second

    mock_build_agent.assert_called_once_with(
        "gpt-4o-mini"
    )



# Test różne modele = różne cache entries

def test_get_agent_builds_separate_agents_for_different_models(
    mocker,
):
    first_agent = mocker.Mock()
    second_agent = mocker.Mock()

    mock_build_agent = mocker.patch(
        "agent.build_agent",
        side_effect=[
            first_agent,
            second_agent,
        ],
    )

    result_1 = agent.get_agent(
        "gpt-4o-mini"
    )

    result_2 = agent.get_agent(
        "gpt-4.1-mini"
    )

    assert result_1 is first_agent
    assert result_2 is second_agent

    assert mock_build_agent.call_count == 2

    assert (
        "gpt-4o-mini"
        in agent._AGENT_CACHE
    )

    assert (
        "gpt-4.1-mini"
        in agent._AGENT_CACHE
    )




# Test niepoprawny model trafia do cache jako default

def test_get_agent_normalizes_model_before_using_cache(
    mocker,
):
    compiled_agent = mocker.Mock()

    mock_build_agent = mocker.patch(
        "agent.build_agent",
        return_value=compiled_agent,
    )

    result = agent.get_agent(
        "invalid-model"
    )

    assert result is compiled_agent

    mock_build_agent.assert_called_once_with(
        agent.DEFAULT_MODEL
    )

    assert (
        agent.DEFAULT_MODEL
        in agent._AGENT_CACHE
    )



# Test clear_agent_cache()

def test_clear_agent_cache(
    mocker,
):
    first_agent = mocker.Mock()
    second_agent = mocker.Mock()

    agent._AGENT_CACHE[
        "gpt-4o-mini"
    ] = first_agent

    agent._AGENT_CACHE[
        "gpt-4.1-mini"
    ] = second_agent

    assert len(
        agent._AGENT_CACHE
    ) == 2

    agent.clear_agent_cache()

    assert (
        agent._AGENT_CACHE
        == {}
    )