const chatContainer = document.getElementById("chatContainer");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const statusText = document.getElementById("status");
const conversationList = document.getElementById("conversationList");
const modelSelect = document.getElementById("modelSelect");
const speechLanguageSelect = document.getElementById("speechLanguageSelect");
const noticeText = document.getElementById("noticeText");
const micBtn = document.getElementById("micBtn");
const toolProgressMap = new Map();

let ragProgress = null;
let waitingForApproval = false;
let threadId = localStorage.getItem("thread_id");

if (!threadId) {
    threadId = crypto.randomUUID();
    localStorage.setItem("thread_id", threadId);
}

const savedModel = localStorage.getItem("selected_model");

if (savedModel && modelSelect) {
    modelSelect.value = savedModel;
}

if (modelSelect) {
    modelSelect.addEventListener("change", () => {
        localStorage.setItem("selected_model", modelSelect.value);

        if (noticeText) {
            noticeText.textContent = `Selected model: ${modelSelect.value}`;
        }
    });
}

let recognition = null;
let isDictating = false;

const savedSpeechLanguage =
    localStorage.getItem("speech_language");

if (savedSpeechLanguage && speechLanguageSelect) {
    speechLanguageSelect.value =
        savedSpeechLanguage;
}

if (speechLanguageSelect) {
    speechLanguageSelect.addEventListener(
        "change",
        () => {

            localStorage.setItem(
                "speech_language",
                speechLanguageSelect.value
            );

            // If recognition already exists,
            // recreate it with the new language.
            if (recognition) {

                if (isDictating) {
                    stopDictation();
                }

                recognition = null;
            }
        }
    );
}

function setupSpeechRecognition() {
    const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        return null;
    }

    const speechRecognition = new SpeechRecognition();

    speechRecognition.lang =
        speechLanguageSelect
            ? speechLanguageSelect.value
            : "pl-PL";
    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;

    speechRecognition.onstart = () => {
        isDictating = true;

        if (micBtn) {
            micBtn.classList.add("recording");
            micBtn.textContent = "⏹️";
        }

        statusText.textContent = "Listening...";
    };

    speechRecognition.onresult = (event) => {
        let finalTranscript = "";
        let interimTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;

            if (event.results[i].isFinal) {
                finalTranscript += transcript + " ";
            } else {
                interimTranscript += transcript;
            }
        }

        if (finalTranscript) {
            const currentText = messageInput.value.trim();

            messageInput.value = currentText
                ? currentText + " " + finalTranscript.trim()
                : finalTranscript.trim();

            autoResize(messageInput);
        }

        if (interimTranscript && noticeText) {
            noticeText.textContent = "Listening: " + interimTranscript;
        }
    };

    speechRecognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);

        if (event.error === "not-allowed") {
            alert("Microphone permission denied. Please allow microphone access.");
        }

        stopDictation();
    };

    speechRecognition.onend = () => {
        if (isDictating) {
            try {
                speechRecognition.start();
            } catch (error) {
                stopDictation();
            }
        }
    };

    return speechRecognition;
}

function toggleDictation() {
    if (!recognition) {
        recognition = setupSpeechRecognition();
    }

    if (!recognition) {
        alert("Speech recognition is not supported in this browser. Please use Chrome or Edge.");
        return;
    }

    if (isDictating) {
        stopDictation();
    } else {
        startDictation();
    }
}

function startDictation() {
    try {
        recognition.start();
    } catch (error) {
        console.error("Could not start dictation:", error);
    }
}

function stopDictation() {
    isDictating = false;

    if (recognition) {
        try {
            recognition.stop();
        } catch (error) {
            console.error("Could not stop dictation:", error);
        }
    }

    if (micBtn) {
        micBtn.classList.remove("recording");
        micBtn.textContent = "🎙️";
    }

    statusText.textContent = "Ready";

    if (noticeText) {
        noticeText.textContent = "RobBotGPT can make mistakes. Check important info.";
    }

    messageInput.focus();
}

function autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
}

function hideWelcome() {
    const welcome = document.getElementById("welcome");
    const cards = document.getElementById("cards");

    if (welcome) welcome.style.display = "none";
    if (cards) cards.style.display = "none";
}

function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function usePrompt(text) {
    messageInput.value = text;
    autoResize(messageInput);
    messageInput.focus();
}

function openFilePicker() {
    document.getElementById("fileInput").click();
}

function addMessage(role, content = "") {
    hideWelcome();

    const messageDiv = document.createElement("div");
    messageDiv.className = role === "user"
        ? "message user"
        : "message assistant";

    const avatar = document.createElement("div");
    avatar.className = role === "user"
        ? "avatar user-avatar"
        : "avatar bot-avatar";

    avatar.textContent = role === "user" ? "U" : "AI";

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.textContent = content;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return messageContent;
}



function addFileMessage(file) {
    hideWelcome();

    const wrapper = document.createElement("div");
    wrapper.className = "file-message";

    const card = document.createElement("div");
    card.className = "file-card";

    // -----------------------------------------
    // File icon
    // -----------------------------------------

    const icon = document.createElement("div");
    icon.className = "file-card-icon";

    icon.innerHTML = `
        <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
        >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
    `;

    // -----------------------------------------
    // File info
    // -----------------------------------------

    const info = document.createElement("div");
    info.className = "file-card-info";

    const name = document.createElement("div");
    name.className = "file-card-name";
    name.textContent = file.name;

    const type = document.createElement("div");
    type.className = "file-card-type";

    const extension =
        file.name.includes(".")
            ? file.name.split(".").pop()
            : "file";

    type.textContent = extension;

    info.appendChild(name);
    info.appendChild(type);

    card.appendChild(icon);
    card.appendChild(info);

    wrapper.appendChild(card);

    chatContainer.appendChild(wrapper);

    chatContainer.scrollTop =
        chatContainer.scrollHeight;
}


function addRagSources(sources) {

    if (!sources || !sources.length) {
        return;
    }

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "rag-sources";

    const title =
        document.createElement("div");

    title.className =
        "rag-sources-title";

    title.textContent =
        "Sources";

    wrapper.appendChild(title);


    sources.forEach(source => {

        const item =
            document.createElement("div");

        item.className =
            "rag-source-item";

        let locationText = "";

        if (source.page) {

            locationText =
                ` — strona ${source.page}`;

        } else if (source.chunk_index) {

            locationText =
                ` — fragment ${source.chunk_index}`;
        }

        item.textContent =
            `📄 ${source.source}${locationText}`;

        wrapper.appendChild(item);
    });


    chatContainer.appendChild(
        wrapper
    );

    chatContainer.scrollTop =
        chatContainer.scrollHeight;
}



function addToolProgress(toolName) {
    const wrapper = document.createElement("div");
    wrapper.className = "tool-progress";

    const box = document.createElement("div");
    box.className = "tool-progress-box";

    const icon = document.createElement("span");
    icon.className = "tool-spinner";

    const text = document.createElement("span");
    text.textContent = `Using ${toolName}...`;

    box.appendChild(icon);
    box.appendChild(text);
    wrapper.appendChild(box);

    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return {
        wrapper,
        icon,
        text
    };
}


function completeToolProgress(toolProgress, toolName) {
    if (!toolProgress) return;

    toolProgress.icon.className = "tool-check";
    toolProgress.icon.textContent = "✓";
    toolProgress.text.textContent = `${toolName} completed`;
}


function formatToolName(toolName) {
    const names = {
        calculator: "Calculator",
        get_current_weather: "Current Weather",
        get_stock_price: "Stock Price",
        purchase_stock: "Purchase Stock",
        remember_this: "Memory Save",
        recall_memory: "Memory Recall",
        tavily_search: "Web Search"
    };

    return names[toolName] || toolName;
}


function addApprovalPrompt(payload) {
    const wrapper = document.createElement("div");
    wrapper.className = "tool-progress";

    const box = document.createElement("div");
    box.className = "tool-progress-box";

    box.style.flexDirection = "column";
    box.style.alignItems = "flex-start";

    const message = document.createElement("div");

    if (typeof payload === "string") {
        message.textContent = payload;
    } else {
        message.textContent =
            payload?.message || "Approval required.";
    }

    const actions = document.createElement("div");
    actions.style.display = "flex";
    actions.style.gap = "8px";
    actions.style.marginTop = "8px";


    const approveBtn = document.createElement("button");
    approveBtn.textContent = "Approve";
    approveBtn.className = "approval-btn";


    const declineBtn = document.createElement("button");
    declineBtn.textContent = "Decline";
    declineBtn.className = "approval-btn";


    approveBtn.onclick = () => {
        approveBtn.disabled = true;
        declineBtn.disabled = true;

        resumeInterruptedGraph(
            true,
            wrapper
        );
    };


    declineBtn.onclick = () => {
        approveBtn.disabled = true;
        declineBtn.disabled = true;

        resumeInterruptedGraph(
            false,
            wrapper
        );
    };


    actions.appendChild(approveBtn);
    actions.appendChild(declineBtn);

    box.appendChild(message);
    box.appendChild(actions);

    wrapper.appendChild(box);

    chatContainer.appendChild(wrapper);

    chatContainer.scrollTop =
        chatContainer.scrollHeight;
}


async function loadConversations() {
    try {
        const response =
            await fetch("/conversations");

        await ensureResponseOk(
            response,
            "Could not load conversations."
        );

        const data =
            await response.json();

        conversationList.innerHTML = "";

        if (!data.conversations || data.conversations.length === 0) {
            conversationList.innerHTML = `
                <div class="history-item">No chats yet</div>
            `;
            return;
        }

        data.conversations.forEach(conv => {

            const item = document.createElement("div");
            item.className = "history-item";


            if (conv.thread_id === threadId) {
                item.classList.add("active");
            }


            // -----------------------------------------
            // Conversation title
            // -----------------------------------------

            const title = document.createElement("span");

            title.className = "history-item-title";

            title.textContent =
                conv.title || "New Chat";


            // -----------------------------------------
            // Delete button
            // -----------------------------------------

            const deleteBtn =
                document.createElement("button");

            deleteBtn.className =
                "delete-chat-btn";

            deleteBtn.textContent = "🗑️";

            deleteBtn.title =
                "Delete conversation";

            deleteBtn.setAttribute(
                "aria-label",
                "Delete conversation"
            );


            // Open conversation
            item.onclick = () => {
                loadConversation(conv.thread_id);
            };


            // Delete conversation
            deleteBtn.onclick = (event) => {

                // Prevent opening conversation
                // when clicking the trash icon
                event.stopPropagation();

                deleteConversation(
                    conv.thread_id
                );
            };


            item.appendChild(title);
            item.appendChild(deleteBtn);

            conversationList.appendChild(item);
        });

    } catch (error) {
        console.error("Failed to load conversations:", error);
    }
}


async function deleteConversation(conversationThreadId) {

    const confirmed = window.confirm(
        "Czy na pewno chcesz usunąć tę konwersację?"
    );

    if (!confirmed) {
        return;
    }


    try {

        const response = await fetch(
            `/conversations/${encodeURIComponent(conversationThreadId)}`,
            {
                method: "DELETE"
            }
        );

        await ensureResponseOk(
            response,
            "Could not delete conversation."
        );

        const data =
            await response.json();

        if (!data.success) {
            throw new Error(
                getApiErrorMessage(
                    data,
                    "Could not delete conversation."
                )
            );
        }
        
        // -----------------------------------------
        // If currently opened conversation
        // was deleted -> create a new empty chat
        // -----------------------------------------

        if (conversationThreadId === threadId) {

            threadId = crypto.randomUUID();

            localStorage.setItem(
                "thread_id",
                threadId
            );

            renderWelcome();
        }


        // Refresh sidebar
        await loadConversations();


    } catch (error) {

        console.error(
            "Failed to delete conversation:",
            error
        );

        alert(
            error.message
        );
    }
}



async function loadConversation(selectedThreadId) {
    threadId = selectedThreadId;
    localStorage.setItem("thread_id", threadId);

    try {
        const response = await fetch(
            `/history/${
                encodeURIComponent(
                    threadId
                )
            }`
        );

        await ensureResponseOk(
            response,
            "Could not load conversation."
        );

        const data =
            await response.json();

        chatContainer.innerHTML = "";

        if (!data.messages || data.messages.length === 0) {
            renderWelcome();
            await loadConversations();
            return;
        }

        data.messages.forEach(msg => {

            addMessage(
                msg.role === "user"
                    ? "user"
                    : "assistant",
                msg.content
            );

            if (
                msg.role === "assistant" &&
                msg.sources &&
                msg.sources.length > 0
            ) {
                addRagSources(
                    msg.sources
                );
            }
        });

        await loadConversations();

    } catch (error) {
        console.error("Failed to load conversation:", error);
    }
}

function renderWelcome() {
    chatContainer.innerHTML = `
        <div class="welcome" id="welcome">
            <h1>How can I help you today?</h1>
            <p>Ask questions, upload documents, use tools, search the web, and chat with memory.</p>
        </div>

        <div class="cards" id="cards">
            <div class="card" onclick="usePrompt('Search the web for latest AI agent news.')">
                Search latest web info
            </div>

            <div class="card" onclick="usePrompt('Summarize the document I uploaded.')">
                Summarize uploaded document
            </div>

            <div class="card" onclick="usePrompt('Remember that my channel name is dswithbappy.')">
                Save something to memory
            </div>

            <div class="card" onclick="usePrompt('Calculate 125 * 48 / 6')">
                Use calculator tool
            </div>
        </div>
    `;
}

function parseSSEPart(part) {
    const lines = part
        .split(/\r?\n/)
        .filter(line => line.trim().startsWith("data:"));

    if (lines.length === 0) {
        return null;
    }

    const jsonText = lines
        .map(line => line.replace(/^data:\s*/, ""))
        .join("\n")
        .trim();

    if (!jsonText || jsonText === "[DONE]") {
        return null;
    }

    try {
        return JSON.parse(jsonText);
    } catch (error) {
        console.error("Invalid stream JSON:", jsonText, error);
        return null;
    }
}


async function consumeSSEStream(
    response,
    botElement,
    selectedModel
) {
    if (!response.body) {
        throw new Error(
            "Streaming is not supported by this browser."
        );
    }

    const reader =
        response.body.getReader();

    const decoder =
        new TextDecoder("utf-8");

    let buffer = "";

    try {
        while (true) {
            const { value, done } =
                await reader.read();

            if (done) {
                break;
            }

            buffer += decoder.decode(
                value,
                { stream: true }
            );

            const parts =
                buffer.split(
                    /\r?\n\r?\n/
                );

            buffer =
                parts.pop() || "";

            for (const part of parts) {
                const data =
                    parseSSEPart(part);

                handleStreamData(
                    data,
                    botElement,
                    selectedModel
                );
            }
        }

        buffer += decoder.decode();

        if (buffer.trim()) {
            const data =
                parseSSEPart(buffer);

            handleStreamData(
                data,
                botElement,
                selectedModel
            );
        }

    } finally {
        reader.releaseLock();
    }
}



async function ensureResponseOk(
    response,
    fallbackMessage = "Request failed."
) {
    if (response.ok) {
        return;
    }

    let data = null;

    try {
        data = await response.json();
    } catch (error) {
        console.error(
            "Could not parse error response:",
            error
        );
    }

    throw new Error(
        getApiErrorMessage(
            data,
            fallbackMessage
        )
    );
}



function getApiErrorMessage(
    data,
    fallbackMessage = "Request failed."
) {
    return (
        data?.message ||
        data?.detail ||
        data?.error ||
        fallbackMessage
    );
}



function handleStreamData(
        data,
        botElement,
        selectedModel
    ) {
        if (!data) return;

        // -----------------------------------------
        // RAG SEARCH START
        // -----------------------------------------

        if (data.type === "rag_search_start") {

            ragProgress =
                addToolProgress("Document Search");

            statusText.textContent =
                "Searching documents...";

            return;
        }

        // -----------------------------------------
        // RAG SEARCH END
        // -----------------------------------------

        if (data.type === "rag_search_end") {

            if (ragProgress) {

                completeToolProgress(
                    ragProgress,
                    "Document Search"
                );

                ragProgress = null;
            }

            statusText.textContent =
                "Document Search completed";

            return;
        }

        if (data.type === "rag_sources") {

            addRagSources(
                data.sources
            );

            return;
        }

        // -----------------------------------------
        // TOOL START
        // -----------------------------------------
        if (data.type === "tool_start") {
            const toolName = formatToolName(data.tool);

            const progress = addToolProgress(toolName);

            const key =
                data.tool_call_id || data.tool;

            toolProgressMap.set(
                key,
                progress
            );

            statusText.textContent =
                `Using ${toolName}...`;

            return;
        }

        // -----------------------------------------
        // TOOL END
        // -----------------------------------------
        if (data.type === "tool_end") {
            const toolName = formatToolName(data.tool);

            const key =
                data.tool_call_id || data.tool;

            const progress =
                toolProgressMap.get(key);

            if (progress) {
                completeToolProgress(
                    progress,
                    toolName
                );

                toolProgressMap.delete(key);
            }

            statusText.textContent =
                `${toolName} completed`;

            return;
        }

        // -----------------------------------------
        // HUMAN-IN-THE-LOOP INTERRUPT
        // -----------------------------------------
        if (data.type === "interrupt") {
            waitingForApproval = true;

            addApprovalPrompt(data.payload);

            statusText.textContent =
                "Waiting for approval...";

            return;
        }


        // -----------------------------------------
        // TOKEN
        // -----------------------------------------
        if (
            data.type === "token" &&
            data.token !== undefined &&
            data.token !== null
        ) {
            botElement.textContent += data.token;

            statusText.textContent =
                `Generating with ${selectedModel}...`;

            chatContainer.scrollTop =
                chatContainer.scrollHeight;

            return;
        }

        // -----------------------------------------
        // ERROR
        // -----------------------------------------
        if (data.type === "error" || data.error) {
            botElement.textContent +=
                "\n\nError: " + data.error;

            statusText.textContent = "Ready";

            return;
        }

        // -----------------------------------------
        // DONE
        // -----------------------------------------
        if (data.type === "done" || data.done) {
            statusText.textContent = "Ready";
        }
    }

async function resumeInterruptedGraph(
    approved,
    approvalElement
) {
    waitingForApproval = false;

    const selectedModel =
        modelSelect
            ? modelSelect.value
            : "gpt-4o-mini";

    statusText.textContent =
        approved
            ? "Purchase approved..."
            : "Purchase declined...";


    const botElement =
        addMessage("assistant", "");


    try {
        const response = await fetch(
            "/chat/stream",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    thread_id: threadId,
                    model: selectedModel,
                    resume: approved
                })
            }
        );


        await ensureResponseOk(
            response,
            "Could not resume agent."
        );

        if (approvalElement) {
            const buttons =
                approvalElement.querySelectorAll(
                    "button"
                );

            buttons.forEach(
                button =>
                    button.disabled = true
            );
        }

        await consumeSSEStream(
            response,
            botElement,
            selectedModel
        );


    } catch (error) {

        console.error(
            "Resume error:",
            error
        );

        botElement.textContent =
            error.message;

    } finally {

        statusText.textContent =
            "Ready";
    }
}    


async function sendMessage() {
    const message = messageInput.value.trim();

    if (!message) return;

    if (isDictating) {
        stopDictation();
    }

    const selectedModel = modelSelect ? modelSelect.value : "gpt-4o-mini";

    addMessage("user", message);

    messageInput.value = "";
    messageInput.style.height = "auto";

    sendBtn.disabled = true;
    statusText.textContent = `Thinking with ${selectedModel}...`;

    toolProgressMap.clear();

    const botElement = addMessage("assistant", "");

    

    try {
        const response = await fetch("/chat/stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                thread_id: threadId,
                model: selectedModel,
                speech_language:
                    speechLanguageSelect
                        ? speechLanguageSelect.value
                        : "pl-PL"
            })
        });

        await ensureResponseOk(
            response,
            "Request failed."
        );

        await consumeSSEStream(
            response,
            botElement,
            selectedModel
        );


    } catch (error) {
        console.error("Streaming error:", error);

        botElement.textContent = error.message;

    } finally {
        sendBtn.disabled = false;
    
        if (!waitingForApproval) {
            statusText.textContent = "Ready";
        }

        messageInput.focus();

        await loadConversations();
    }
}

async function uploadFile() {
    const fileInput = document.getElementById("fileInput");

    if (!fileInput.files.length) {
        return;
    }

    const file = fileInput.files[0];

    addFileMessage(file);

    const toolProgress = addToolProgress("Document Ingestion");
    statusText.textContent = "Using Document Ingestion...";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("thread_id", threadId);

    try {
        const response = await fetch(
            "/upload",
            {
                method: "POST",
                body: formData
            }
        );

        await ensureResponseOk(
            response,
            "Upload failed."
        );

        const data =
            await response.json();

        if (!data.success) {
            throw new Error(
                getApiErrorMessage(
                    data,
                    "Upload failed."
                )
            );
        }

        completeToolProgress(
            toolProgress,
            "Document Ingestion"
        );

        await loadConversations();


    } catch (error) {

        completeToolProgress(
            toolProgress,
            "Document Ingestion"
        );

        console.error(
            "Upload error:",
            error
        );

        addMessage(
            "assistant",
            "Upload failed: "
            + error.message
        );
    }

    statusText.textContent = "Ready";
    fileInput.value = "";
}

async function newChat() {
    threadId = crypto.randomUUID();
    localStorage.setItem("thread_id", threadId);

    if (isDictating) {
        stopDictation();
    }

    renderWelcome();
    await loadConversations();
    messageInput.focus();
}

loadConversations();

if (threadId) {
    loadConversation(threadId);
}