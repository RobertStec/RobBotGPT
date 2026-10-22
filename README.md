# RobBotGPT

**RobBotGPT** is a production-deployed AI agent application built with **LangGraph, LangChain, FastAPI, PostgreSQL and pgvector**.

The project combines conversational AI, tool calling, Retrieval-Augmented Generation, persistent LangGraph state, long-term memory, Human-in-the-Loop workflows, LangSmith observability, automated evaluation, CI/CD and cloud deployment.

**Live application:** https://robbotgpt.aispectra.pl

---

## Project Overview

RobBotGPT was created as a practical AI engineering project focused on building and deploying a complete agentic application rather than a simple LLM wrapper.

The application supports:

- conversational AI with streaming responses,
- LangGraph-based agent orchestration,
- tool calling,
- Retrieval-Augmented Generation,
- query rewriting,
- document retrieval with pgvector,
- reranking,
- relevance filtering,
- RAG fallback when the available context is insufficient,
- persistent conversation state,
- long-term memory,
- Human-in-the-Loop approval,
- document uploads,
- LangSmith tracing and evaluation,
- automated regression testing,
- Docker-based deployment,
- GitHub Actions CI/CD,
- production deployment on AWS.

---

## Key Features

### LangGraph Agent

The core application is orchestrated with LangGraph.

The graph is responsible for routing requests between:

- normal conversational responses,
- tools,
- document-based RAG,
- memory operations,
- Human-in-the-Loop workflows.

LangGraph state is persisted in PostgreSQL using:

```text
langgraph-checkpoint-postgres
```

This allows conversations to survive application restarts and deployments.

---


### LangGraph Workflow

```mermaid
flowchart TD
    START([START]) --> ROUTER[router]

    ROUTER -->|chatbot| CHATBOT[chatbot]
    ROUTER -->|document| REWRITE[rewrite_query]

    REWRITE --> RETRIEVE[retrieve]
    RETRIEVE --> RERANK[rerank]

    RERANK -->|Relevant context| GENERATE[generate]
    RERANK -->|Insufficient context| NOANSWER[no_answer]

    GENERATE --> END1([END])
    NOANSWER --> END2([END])

    CHATBOT -->|Tool call| TOOLS[tools]
    TOOLS --> CHATBOT
    CHATBOT -->|No tool call| END3([END])
```

The router separates normal agent conversations from document-based RAG requests.

The chatbot branch can dynamically execute tools and return to the chatbot until the agent produces a final response.

The document branch performs query rewriting, retrieval and reranking before routing either to answer generation or to a controlled no-answer fallback.


### Advanced RAG Pipeline

RobBotGPT contains a multi-stage Retrieval-Augmented Generation pipeline.

```mermaid
flowchart LR
    Q[User Question]
    RW[rewrite_query]
    RET[retrieve]
    RR[rerank]
    GEN[generate]
    NA[no_answer]

    Q --> RW
    RW --> RET
    RET --> RR

    RR -->|Relevant context| GEN
    RR -->|Insufficient context| NA

    GEN --> END1[END]
    NA --> END2[END]
```

The RAG system includes:

- query rewriting using conversation context,
- semantic retrieval,
- PostgreSQL with pgvector,
- configurable retrieval candidate count,
- reranking,
- relevance threshold,
- final context selection,
- source tracking,
- document isolation by conversation,
- explicit fallback instead of hallucinating answers.

Uploaded documents are split into chunks and stored together with vector embeddings.

Supported document types include:

```text
PDF
DOCX
TXT
Markdown
Python
CSV
```

---

## System Architecture

```mermaid
flowchart LR
    USER[User / Browser]

    API[FastAPI]

    GRAPH[LangGraph Agent]

    TOOLS[Agent Tools]
    RAG[RAG Pipeline]
    MEMORY[Memory]
    HITL[Human-in-the-Loop]

    PG[(PostgreSQL + pgvector)]

    OPENAI[OpenAI]
    EXTERNAL[External APIs]
    LANGSMITH[LangSmith]

    USER --> API
    API --> GRAPH

    GRAPH --> TOOLS
    GRAPH --> RAG
    GRAPH --> MEMORY
    GRAPH --> HITL

    GRAPH --> OPENAI

    TOOLS --> EXTERNAL

    GRAPH --> PG
    RAG --> PG
    MEMORY --> PG

    GRAPH --> LANGSMITH
```

---

## Agent Tools

RobBotGPT supports multiple tools that can be dynamically selected by the agent.

Examples include:

- calculator,
- weather lookup,
- stock price lookup,
- Tavily web search,
- long-term memory save,
- long-term memory recall,
- stock purchase workflow with Human-in-the-Loop approval.

The calculator uses safe AST-based expression evaluation instead of Python `eval()`.

External API integrations are configured through environment variables.

---

## Human-in-the-Loop

Selected actions can interrupt the LangGraph workflow and require explicit confirmation from the user.

Example:

```text
User
  ↓
Agent decides to execute sensitive action
  ↓
LangGraph interrupt()
  ↓
Frontend displays Approve / Decline
  ↓
User decision
  ↓
Command(resume=...)
  ↓
Graph continues execution
```

This demonstrates controlled agent execution for operations that should not be performed automatically.

---

## Memory and Persistence

RobBotGPT supports two different persistence mechanisms.

### Conversation state

LangGraph checkpoints are stored in PostgreSQL.

Production tables include:

```text
checkpoints
checkpoint_blobs
checkpoint_writes
checkpoint_migrations
```

This allows graph state to survive container and application restarts.

### Application memory

The application also stores:

- conversations,
- chat messages,
- long-term memory,
- RAG documents,
- RAG chunks.

Persistent state was verified on the production environment by restarting the application container and continuing the same conversation afterwards.

---

## PostgreSQL and pgvector

PostgreSQL is used as the main application database.

The project uses:

```text
PostgreSQL 16
pgvector
SQLAlchemy
psycopg
Alembic
```

Database migrations are managed with Alembic.

Main database areas include:

```text
conversations
chat_messages
long_term_memory
rag_documents
rag_chunks
LangGraph checkpoints
```

---

## LangSmith and Evaluation

LangSmith is used for tracing and observing LangGraph executions.

The project also contains a dedicated evaluation layer:

```text
evals/
├── data/
├── evaluators.py
├── langsmith_client.py
├── regression.py
├── run_experiment.py
├── schemas.py
├── sync_dataset.py
├── target.py
└── tracing.py
```

The evaluation system supports:

- LangSmith datasets,
- RAG evaluation cases,
- custom evaluators,
- regression detection,
- automated quality gates.

Regression evaluation is integrated into GitHub Actions and can block Pull Requests when quality requirements are not met.

---

## Testing

The project contains both unit and integration tests.

```text
tests/
├── integration/
└── unit/
```

Tests cover areas such as:

- LangGraph agent construction,
- graph nodes,
- routing,
- tools,
- streaming,
- RAG,
- reranking,
- configuration,
- database layer,
- checkpoints,
- conversations,
- document services,
- API endpoints,
- PostgreSQL integration,
- Human-in-the-Loop,
- evaluation pipeline,
- regression evaluation.

Tests are executed with:

```bash
pytest
```

Development dependencies are defined in:

```text
requirements-dev.txt
```

---

## CI/CD

The repository contains four GitHub Actions workflows:

```text
.github/workflows/
├── ci.yml
├── regression.yml
├── docker-publish.yml
└── deploy.yml
```

### Pull Request Quality Gate

Pull Requests run automated checks including:

```text
Python tests
PostgreSQL integration tests
Docker build
Dependency security audit
Regression Quality Gate
```

A Pull Request can only be merged after required checks pass.

### Deployment Pipeline

After changes are merged into `main`:

```mermaid
flowchart LR
    PR[Pull Request]
    CI[CI]
    REG[Regression Quality Gate]
    MAIN[Merge to main]
    IMG[Docker Image Publish]
    AWS[Deploy to AWS]
    PROD[Production Application]

    PR --> CI
    PR --> REG
    CI --> MAIN
    REG --> MAIN
    MAIN --> IMG
    IMG --> AWS
    AWS --> PROD
```

The deployment pipeline publishes a Docker image and deploys the new version to the production server.

---

## Production Deployment

RobBotGPT is deployed on AWS.

Production infrastructure includes:

```text
AWS EC2
Docker
Docker Compose
PostgreSQL + pgvector
Nginx
Let's Encrypt
HTTPS
GitHub Actions
Docker image registry
```

The application is exposed through Nginx while the application container itself listens internally on port `8080`.

Production health checks monitor both:

- application container,
- PostgreSQL container.

Live deployment:

```text
https://robbotgpt.aispectra.pl
```

---

## Docker

The application uses a multi-stage Docker build based on:

```text
python:3.11-slim
```

The runtime image:

- runs as a non-root user,
- drops Linux capabilities,
- prevents privilege escalation,
- uses a read-only filesystem,
- provides a writable temporary filesystem only where required,
- includes health checks.

### Start the full stack

Create your environment file first:

```bash
cp .env.example .env
```

Configure required API keys and database settings.

Then run:

```bash
docker compose up --build
```

The application will be available at:

```text
http://localhost:8080
```

PostgreSQL is exposed locally on:

```text
localhost:5433
```

Stop the stack with:

```bash
docker compose down
```

---

## Local Development

### 1. Clone the repository

```bash
git clone https://github.com/RobertStec/RobBotGPT.git
cd RobBotGPT
```

### 2. Create a Python environment

Using Conda:

```bash
conda create -n robbotgpt python=3.11 -y
conda activate robbotgpt
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

Copy:

```bash
cp .env.example .env
```

Then configure the required values.

Do not commit `.env` to Git.

### 5. Start PostgreSQL

You can use Docker Compose:

```bash
docker compose up -d db
```

Make sure the database credentials in `.env` match your local PostgreSQL configuration.

### 6. Run migrations

```bash
alembic upgrade head
```

### 7. Start the application

```bash
python app.py
```

or:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Open:

```text
http://localhost:8080
```

---

## Environment Variables

The repository contains:

```text
.env.example
```

Main configuration groups include:

### OpenAI

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### External tools

```env
TAVILY_API_KEY=
ALPHA_VANTAGE_API_KEY=
OPENWEATHER_API_KEY=
```

### LangSmith

```env
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=RobBotGPT
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

### RAG

```env
RAG_CHUNK_SIZE=900
RAG_CHUNK_OVERLAP=150
RAG_TOP_K=4

RAG_RETRIEVAL_TOP_K=12
RAG_FINAL_TOP_K=4

RAG_RELEVANCE_THRESHOLD=0.55
RAG_CONTEXT_MAX_CHARS=6000
```

### Application

```env
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8080
APP_RELOAD=true

LOG_LEVEL=INFO
```

### PostgreSQL

Example:

```env
POSTGRES_DB=robbotgpt
POSTGRES_USER=robbotgpt
POSTGRES_PASSWORD=change_me
POSTGRES_HOST=localhost
POSTGRES_PORT=5433

DATABASE_URL=postgresql+psycopg://robbotgpt:change_me@localhost:5433/robbotgpt
```

Never store real production credentials in the repository.

---

## Database Migrations

Database schema changes are managed through Alembic.

Apply migrations:

```bash
alembic upgrade head
```

Current migrations include the initial application schema and pgvector-based RAG storage.

---

## Project Structure

```text
RobBotGPT/
│
├── api/
│   └── routers/
│       ├── chat.py
│       ├── conversations.py
│       └── documents.py
│
├── core/
│   ├── config.py
│   ├── environment.py
│   ├── exceptions.py
│   ├── lifecycle.py
│   └── logging.py
│
├── services/
│   ├── conversation_service.py
│   ├── document_service.py
│   └── streaming.py
│
├── evals/
│   ├── data/
│   ├── evaluators.py
│   ├── langsmith_client.py
│   ├── regression.py
│   ├── run_experiment.py
│   ├── schemas.py
│   ├── sync_dataset.py
│   ├── target.py
│   └── tracing.py
│
├── tests/
│   ├── integration/
│   └── unit/
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── index.html
│
├── alembic/
│   └── versions/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── deploy.yml
│       ├── docker-publish.yml
│       └── regression.yml
│
├── agent.py
├── app.py
├── checkpoints.py
├── database.py
├── nodes.py
├── rag.py
├── reranker.py
├── state.py
├── tools.py
│
├── Dockerfile
├── docker-compose.yml
├── alembic.ini
│
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
│
├── .env.example
├── README.md
└── LICENSE
```

---

## Reliability and Security

The project includes several reliability and security mechanisms:

- environment-based configuration,
- centralized logging,
- application health checks,
- PostgreSQL health checks,
- structured exception handling,
- non-root Docker runtime,
- read-only application container,
- dropped Linux capabilities,
- `no-new-privileges`,
- dependency security auditing,
- automated tests,
- Pull Request quality gates,
- RAG fallback instead of unsupported answers,
- Human-in-the-Loop for selected actions,
- safe calculator implementation without `eval()`.

---

## Example Use Cases

### General agent conversation

```text
Explain how LangGraph differs from a traditional LangChain agent.
```

### Document RAG

Upload a document and ask:

```text
Answer based on the uploaded document:
What is the main color of the Atlas project interface?
```

RobBotGPT retrieves relevant chunks and provides the answer together with document sources.

### Missing document information

If the requested information does not exist in the uploaded documents, the application returns a controlled fallback instead of inventing an answer.

### Long-term memory

```text
Remember that my project code is KAPPA-7319.
```

Later:

```text
What project code did I give you?
```

### Tool usage

```text
Calculate 125 * 48 / 6.
```

or:

```text
What is the current weather in Warsaw?
```

### Human-in-the-Loop

Selected actions can pause execution and require explicit approval before the graph continues.

---

## Technology Stack

### Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic Settings
- SQLAlchemy
- Alembic

### AI / Agents

- LangChain
- LangGraph
- OpenAI
- LangSmith
- Tavily

### RAG

- OpenAI embeddings
- PostgreSQL
- pgvector
- custom retrieval pipeline
- query rewriting
- reranking
- relevance filtering

### Infrastructure

- Docker
- Docker Compose
- GitHub Actions
- AWS EC2
- Nginx
- Let's Encrypt

### Testing

- pytest
- unit tests
- integration tests
- PostgreSQL integration tests
- regression evaluation
- LangSmith datasets and evaluators

---

## Future Improvements

Possible future extensions include:

- authentication and user accounts,
- rate limiting,
- distributed task processing,
- Redis caching,
- multi-user document isolation,
- additional agent tools,
- enhanced observability,
- horizontal scaling,
- infrastructure as code,
- Kubernetes deployment.

---

## License

This project includes a `LICENSE` file. See it for licensing details.

---

## Author

**Robert Stec**

RobBotGPT was developed as a portfolio project focused on practical AI engineering, LangGraph agent architecture, RAG systems, evaluation, testing and production deployment.