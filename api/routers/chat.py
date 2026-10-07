import logging

from fastapi import (
    APIRouter,
    Request,
)
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Command

from agent import get_agent

from database import (
    create_or_update_conversation,
    save_chat_message,
)

from services.streaming import (
    extract_text_from_chunk,
    should_stream_chunk,
    sse_data,
)


logger = logging.getLogger(__name__)

router = APIRouter()



@router.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body."},
            status_code=400
        )

    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", "gpt-4o-mini")
    speech_language = data.get("speech_language", "pl-PL")

    is_resume = "resume" in data
    resume_value = data.get("resume")

    if not is_resume and not user_message.strip():
        return JSONResponse(
            {"error": "Message is required."},
            status_code=400
        )

    agent = get_agent(selected_model)

    if not is_resume:
        create_or_update_conversation(
            thread_id,
            user_message
        )

        save_chat_message(
            thread_id,
            "user",
            user_message
        )


    config = {
        "configurable": {
            "thread_id": thread_id
        },

        "run_name": "RobBotGPT",

        "tags": [
            "robbotgpt",
            "langgraph"
        ],

        "metadata": {
            "thread_id": thread_id,
            "model": selected_model,
            "speech_language": speech_language
        }
    }

    def event_generator():
        final_answer = ""

        rag_sources = []

        # Zapamiętujemy nazwę narzędzia dla danego tool_call_id
        tool_names = {}

        try:
            if is_resume:
                graph_input = Command(
                    resume=resume_value
                )
            else:
                graph_input = {
                    "messages": [
                        HumanMessage(content=user_message)
                    ],
                    "selected_model": selected_model,
                    "speech_language": speech_language,
                }

            for part in agent.stream(
                graph_input,
                config=config,
                stream_mode=[
                    "messages",
                    "updates",
                    "values",
                    "custom"
                ],
                version="v2"
            ):

                # ==========================================
                # HUMAN-IN-THE-LOOP INTERRUPT
                # ==========================================
                
                interrupts = part.get("interrupts", ()) or ()

                if interrupts:
                    for interrupt_item in interrupts:

                        interrupt_value = getattr(
                            interrupt_item,
                            "value",
                            None
                        )

                        interrupt_id = getattr(
                            interrupt_item,
                            "id",
                            None
                        )

                        yield sse_data({
                            "type": "interrupt",
                            "interrupt_id": interrupt_id,
                            "payload": interrupt_value
                        })

                    continue
                
                part_type = part.get("type")
                part_data = part.get("data")


                # =====================================================
                # CUSTOM - własne zdarzenia workflow
                # =====================================================

                if part_type == "custom":

                    if not isinstance(part_data, dict):
                        continue

                    custom_type = part_data.get("type")

                    if custom_type == "rag_sources":

                        rag_sources = (
                            part_data.get(
                                "sources",
                                []
                            )
                            or []
                        )

                        yield sse_data(part_data)

                        continue


                    if custom_type in {
                        "rag_search_start",
                        "rag_search_end",
                    }:
                        yield sse_data(part_data)

                    continue


                # =====================================================
                # UPDATES - wykrywanie rzeczywistych wywołań narzędzi
                # =====================================================
                if part_type == "updates":

                    if not isinstance(part_data, dict):
                        continue

                    for node_name, node_update in part_data.items():

                        if not isinstance(node_update, dict):
                            continue

                        messages = node_update.get("messages", [])

                        if not isinstance(messages, list):
                            messages = [messages]

                        # ---------------------------------------------
                        # CHATBOT wybrał narzędzie
                        # ---------------------------------------------
                        if node_name in {
                            "chatbot",
                            "document_request"
                        }:

                            for message in messages:

                                if not isinstance(message, AIMessage):
                                    continue

                                tool_calls = getattr(
                                    message,
                                    "tool_calls",
                                    []
                                ) or []

                                for tool_call in tool_calls:

                                    tool_name = tool_call.get("name")
                                    tool_call_id = tool_call.get("id")

                                    if not tool_name:
                                        continue

                                    if tool_call_id:
                                        tool_names[tool_call_id] = tool_name

                                    yield sse_data({
                                        "type": "tool_start",
                                        "tool": tool_name,
                                        "tool_call_id": tool_call_id
                                    })

                        # ---------------------------------------------
                        # TOOL zakończył działanie
                        # ---------------------------------------------
                        elif node_name == "tools":

                            for message in messages:

                                if not isinstance(message, ToolMessage):
                                    continue

                                tool_call_id = getattr(
                                    message,
                                    "tool_call_id",
                                    None
                                )

                                tool_name = getattr(
                                    message,
                                    "name",
                                    None
                                )

                                if not tool_name and tool_call_id:
                                    tool_name = tool_names.get(
                                        tool_call_id
                                    )

                                if not tool_name:
                                    tool_name = "tool"

                                yield sse_data({
                                    "type": "tool_end",
                                    "tool": tool_name,
                                    "tool_call_id": tool_call_id
                                })

                    continue

                # =====================================================
                # MESSAGES - streaming odpowiedzi LLM token po tokenie
                # =====================================================
                if part_type == "messages":

                    if (
                        not isinstance(part_data, tuple)
                        or len(part_data) != 2
                    ):
                        continue

                    chunk, metadata = part_data

                    if not should_stream_chunk(
                        chunk,
                        metadata
                    ):
                        continue

                    token = extract_text_from_chunk(chunk)

                    if token:
                        final_answer += token

                        yield sse_data({
                            "type": "token",
                            "token": token
                        })

            # =========================================================
            # Zapis końcowej odpowiedzi do historii
            # =========================================================
            if final_answer.strip():
                save_chat_message(
                    thread_id,
                    "assistant",
                    final_answer,
                    sources=rag_sources
                )

            yield sse_data({
                "type": "done",
                "done": True
            })

        except Exception:

            logger.exception(
                (
                    "Chat stream failed "
                    "thread_id=%s model=%s"
                ),
                thread_id,
                selected_model,
            )

            yield sse_data({
                "type": "error",
                "error": (
                    "An internal error occurred "
                    "while generating the response."
                ),
                "error_code": (
                    "chat_stream_failed"
                ),
            })

            yield sse_data({
                "type": "done",
                "done": True,
            })


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )