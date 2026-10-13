from collections.abc import Callable

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI

from core.config import settings
from nodes import (
    generate_rag_node,
    no_answer_node,
    rerank_node,
    retrieve_node,
    rewrite_query_node,
    route_after_rerank,
)






def _noop_writer(_event: dict) -> None:
    """
    Ignore streaming events during evaluation.
    """





def build_eval_messages(
    inputs: dict,
) -> list:
    """
    Convert LangSmith-style evaluation inputs
    into LangChain messages.
    """

    messages = []

    history = inputs.get(
        "history",
        [],
    )

    for item in history:

        role = item.get("role")
        content = item.get("content", "")

        if role == "user":
            messages.append(
                HumanMessage(
                    content=content
                )
            )

        elif role == "assistant":
            messages.append(
                AIMessage(
                    content=content
                )
            )

    messages.append(
        HumanMessage(
            content=inputs["question"]
        )
    )

    return messages





def build_rag_eval_target(
    thread_id: str,
    llm=None,
) -> Callable[[dict], dict]:
    """
    Build an evaluation target compatible with
    the input/output style used by LangSmith.

    The thread_id identifies the document corpus
    used during the evaluation.
    """

    if not thread_id.strip():
        raise ValueError(
            "thread_id cannot be empty."
        )

    if llm is None:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,
            api_key=settings.openai_api_key,
        )

    def target(
        inputs: dict,
    ) -> dict:

        state = {
            "messages": build_eval_messages(
                inputs
            )
        }

        # -----------------------------------------
        # Query rewriting
        # -----------------------------------------

        rewrite_result = (
            rewrite_query_node(
                state=state,
                llm=llm,
            )
        )

        state.update(
            rewrite_result
        )

        # -----------------------------------------
        # Retrieval
        # -----------------------------------------

        retrieve_result = retrieve_node(
            state=state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                }
            },
            writer=_noop_writer,
        )

        state.update(
            retrieve_result
        )

        # -----------------------------------------
        # Reranking + relevance filter
        # -----------------------------------------

        rerank_result = rerank_node(
            state=state,
            llm=llm,
        )

        state.update(
            rerank_result
        )

        route = route_after_rerank(
            state
        )

        # -----------------------------------------
        # Final response
        # -----------------------------------------

        if route == "generate":

            response_result = (
                generate_rag_node(
                    state=state,
                    llm=llm,
                    writer=_noop_writer,
                )
            )

        else:

            response_result = (
                no_answer_node(
                    state
                )
            )

        response = (
            response_result["messages"][-1]
        )

        answer = (
            response.content
            if isinstance(
                response.content,
                str,
            )
            else str(response.content)
        )

        return {
            "answer": answer,
            "route": route,
            "sources": state.get(
                "rag_sources",
                [],
            ),
            "rewritten_query": state.get(
                "rag_query",
                "",
            ),
        }

    return target