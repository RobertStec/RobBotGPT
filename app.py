from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import json
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from langgraph.types import Command

from agent import (
    get_agent,
    delete_thread_checkpoints
)

from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations,
    delete_conversation)

from rag import (
    add_document_to_rag,
    delete_thread_documents
)



app = FastAPI()

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


init_db()


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )



@app.get("/conversations")
async def conversations():
    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }



@app.delete("/conversations/{thread_id}")
async def delete_conversation_endpoint(thread_id: str):

    try:
        # -----------------------------------------
        # 1. Delete RAG documents and uploaded files
        # -----------------------------------------

        rag_result = delete_thread_documents(
            thread_id
        )

        # -----------------------------------------
        # 2. Delete LangGraph checkpoints
        # -----------------------------------------

        delete_thread_checkpoints(
            thread_id
        )

        # -----------------------------------------
        # 3. Delete conversation database data
        # -----------------------------------------

        deleted = delete_conversation(
            thread_id
        )

        if not deleted:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Conversation not found."
                },
                status_code=404
            )

        return {
            "success": True,
            "message": "Conversation deleted.",
            "deleted_chunks": rag_result["deleted_chunks"],
            "deleted_files": rag_result["deleted_files"]
        }

    except Exception as e:

        return JSONResponse(
            {
                "success": False,
                "message": (
                    "Could not completely delete conversation: "
                    + str(e)
                )
            },
            status_code=500
        )



@app.get("/history/{thread_id}")
async def history(thread_id: str):
    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }




@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        create_or_update_conversation(thread_id, "Uploaded document")

        result = add_document_to_rag(
            file_path=file_path,
            thread_id=thread_id,
            original_filename=filename
        )

        return JSONResponse({
            "success": True,
            "filename": filename,
            "chunks": result["chunks"]
        })

    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )



def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"



def should_stream_chunk(chunk, metadata) -> bool:
    """
    This prevents raw tool/search/RAG JSON from appearing in the frontend.

    We only stream normal AI text chunks.
    We do NOT stream:
    - ToolMessage
    - messages from tool nodes
    - tool call chunks
    - raw tool outputs
    """

    metadata = metadata or {}

    node_name = str(metadata.get("langgraph_node", "")).lower()

    if "tool" in node_name:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "tool_call_chunks", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}

    if additional_kwargs.get("tool_calls"):
        return False

    return True



def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "".join(text_parts)

    return ""



@app.post("/chat/stream")
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

                    if custom_type in {
                        "rag_search_start",
                        "rag_search_end",
                        "rag_sources",
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
                    final_answer
                )

            yield sse_data({
                "type": "done",
                "done": True
            })

        except Exception as e:

            yield sse_data({
                "type": "error",
                "error": str(e)
            })

            yield sse_data({
                "type": "done",
                "done": True
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





if __name__ == "__main__":
   
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )