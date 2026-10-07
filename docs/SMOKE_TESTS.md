# RobBotGPT — Smoke Test Checklist

## Environment

- Application starts without traceback
- FastAPI lifespan initializes correctly
- Home page loads
- CSS loads from `/static/css/app.css`
- JavaScript loads from `/static/js/app.js`

## Chat

- [x] Simple message returns streamed response
- [x] User message is saved in conversation history
- [x] Assistant response is saved in conversation history
- [x] Conversation appears in sidebar
- [x] Existing conversation can be reopened
- [x] Selected conversation loads correct history

## Tools

- [x] Calculator tool emits tool_start
- [x] Calculator tool emits tool_end
- [x] Tool result is followed by assistant response
- [x] Tool progress indicator completes

Test prompt:

`Calculate 125 * 48 / 6`

## RAG

- [x] TXT/PDF document can be uploaded
- [x] Upload creates conversation
- [x] Document ingestion indicator completes
- [x] Question about uploaded document triggers RAG
- [x] RAG search indicator is displayed
- [x] Assistant returns answer based on document
- [x] Sources are displayed
- [x] Sources remain visible after reopening conversation
- [x] Unsupported file type returns safe error

## HITL

- [x] Stock purchase triggers interrupt
- [x] Approval UI appears
- [x] Approve resumes the same LangGraph thread
- [x] Decline resumes the same LangGraph thread
- [x] No duplicate user message is created after resume
- [x] Final assistant response is stored

Test prompt:

`Buy 5 shares of AAPL`

## Conversations

- [x] New Chat creates a new thread_id
- [x] Existing conversations appear in sidebar
- [x] Conversation can be opened
- [x] Conversation can be deleted
- [x] Deleted conversation disappears from sidebar
- [x] Deleting current conversation creates a new empty chat

## Frontend

- [x] No JavaScript errors in browser console
- [x] Streaming tokens appear correctly
- [x] Send button is disabled while processing
- [x] Status changes correctly
- [x] Error messages from backend are displayed
- [x] Upload progress does not remain stuck
- [x] HITL buttons become disabled after selection
- [x] Model selection persists after reload
- [x] Speech language selection persists after reload

## Error handling

- [x] Invalid upload does not expose traceback
- [x] API errors display safe messages
- [x] Streaming error ends with `done`
- [x] Application remains usable after an error

## Final result

Status: PASS

Date: 07-10-2026
Tester: Robert
Notes:
All core application flows were tested manually.
Chat streaming, tools, RAG, HITL, conversation history,
conversation deletion and frontend error handling worked correctly.
No critical JavaScript errors were observed in DevTools.
Unsupported file uploads were rejected safely without exposing internal errors or tracebacks.