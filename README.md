# RobBotGPT

**RobBotGPT** is a production-deployed AI agent application built with **LangGraph, LangChain, FastAPI, PostgreSQL, pgvector and Redis**.

The project demonstrates how to design, test, secure and deploy a complete agentic AI application rather than a simple LLM wrapper.

It combines:

- conversational AI,
- LangGraph orchestration,
- tool calling,
- Retrieval-Augmented Generation,
- query rewriting and reranking,
- persistent graph state,
- long-term memory,
- Human-in-the-Loop workflows,
- multi-user authentication and isolation,
- secure document uploads,
- LangSmith observability and evaluation,
- automated testing,
- security quality gates,
- Docker-based infrastructure,
- CI/CD,
- production deployment on AWS.

**Live application:** [robbotgpt.aispectra.pl](https://robbotgpt.aispectra.pl)

---

## Project Overview

RobBotGPT was created as a practical AI engineering portfolio project focused on the full lifecycle of an agentic application:

```text
design
  ↓
LangGraph orchestration
  ↓
RAG and tools
  ↓
persistence
  ↓
multi-user security
  ↓
testing and evaluation
  ↓
CI/CD
  ↓
production deployment
```

The application supports:

- streaming conversational AI,
- LangGraph-based routing,
- dynamic tool execution,
- Retrieval-Augmented Generation,
- query rewriting,
- semantic retrieval with pgvector,
- reranking,
- relevance filtering,
- controlled RAG fallback,
- persistent conversation state,
- long-term memory,
- document uploads,
- user authentication and sessions,
- multi-user data isolation,
- Human-in-the-Loop approval,
- LangSmith tracing,
- automated regression evaluation,
- Redis-based rate limiting,
- Docker deployment,
- automated CI/CD,
- production deployment on AWS.

---

## Key Features

### LangGraph Agent

The main AI workflow is orchestrated with **LangGraph**.

The graph routes requests between:

- conversational responses,
- agent tools,
- document-based RAG,
- long-term memory,
- Human-in-the-Loop workflows.

LangGraph state is persisted in PostgreSQL using:

```text
langgraph-checkpoint-postgres
```

This allows graph execution state to survive application restarts and production deployments.

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

The chatbot branch can execute tools and return to the chatbot until the agent produces a final answer.

The document branch performs:

```text
query rewriting
      ↓
semantic retrieval
      ↓
reranking
      ↓
relevance filtering
      ↓
generation / controlled fallback
```

---

## Advanced RAG Pipeline

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
```

The RAG pipeline includes:

- conversation-aware query rewriting,
- OpenAI embeddings,
- PostgreSQL + pgvector,
- semantic similarity search,
- configurable retrieval candidate count,
- reranking,
- relevance thresholds,
- final context selection,
- document source tracking,
- user-level document isolation,
- conversation-level document isolation,
- explicit fallback instead of hallucinated answers.

Uploaded documents are split into chunks and stored together with vector embeddings.

Supported formats include:

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
    USER[Browser / User]

    API[FastAPI API]
    AUTH[Authentication]

    GRAPH[LangGraph Agent]

    TOOLS[Agent Tools]
    RAG[RAG Pipeline]
    MEMORY[Long-Term Memory]
    HITL[Human-in-the-Loop]

    PG[(PostgreSQL + pgvector)]
    REDIS[(Redis)]

    OPENAI[OpenAI]
    EXTERNAL[External APIs]
    LANGSMITH[LangSmith]

    USER --> API
    API --> AUTH
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
    AUTH --> PG

    API --> REDIS

    GRAPH --> LANGSMITH
```

---

## Agent Tools

RobBotGPT supports tools dynamically selected by the agent.

Examples include:

- calculator,
- weather lookup,
- stock price lookup,
- Tavily web search,
- long-term memory storage,
- long-term memory recall,
- stock purchase workflow with Human-in-the-Loop approval.

The calculator uses safe AST-based expression evaluation instead of Python `eval()`.

External API integrations are configured through environment variables.

---

## Human-in-the-Loop

Selected actions can interrupt LangGraph execution and require explicit user confirmation.

Example:

```text
User
  ↓
Agent selects sensitive operation
  ↓
LangGraph interrupt()
  ↓
Frontend displays Approve / Decline
  ↓
User decision
  ↓
Command(resume=...)
  ↓
Graph resumes
```

This demonstrates controlled agent execution for operations that should not be performed automatically.

---

## Authentication and Multi-User Isolation

RobBotGPT supports user accounts and authenticated sessions.

Authentication features include:

- user registration,
- login,
- logout,
- `/auth/me`,
- password hashing,
- session persistence,
- session expiration,
- invalid session rejection,
- inactive user protection,
- HTTP-only session cookies,
- SameSite protection,
- Secure cookies in production.

Application data is scoped by `user_id`.

Isolation applies to:

- conversations,
- chat history,
- long-term memory,
- LangGraph execution,
- uploaded documents,
- RAG documents,
- vector retrieval,
- document deletion.

Attempts to access another user's resources return controlled errors without exposing whether the resource belongs to another account.

---

## Memory and Persistence

RobBotGPT uses multiple persistence mechanisms.

### LangGraph checkpoints

Graph execution state is stored in PostgreSQL.

Production checkpoint tables include:

```text
checkpoints
checkpoint_blobs
checkpoint_writes
checkpoint_migrations
```

This allows graph execution to survive container restarts.

### Application persistence

PostgreSQL also stores:

- users,
- authentication sessions,
- conversations,
- chat messages,
- long-term memories,
- RAG documents,
- RAG chunks.

Persistence was tested both locally and against PostgreSQL integration environments.

---

## PostgreSQL and pgvector

PostgreSQL is the main persistence layer.

The application uses:

```text
PostgreSQL 16
pgvector
SQLAlchemy
psycopg
Alembic
```

Database migrations are managed with Alembic.

The schema includes areas for:

```text
users
auth_sessions
conversations
chat_messages
long_term_memory
rag_documents
rag_chunks
LangGraph checkpoints
```

User ownership is enforced at the application and persistence layers.

---

## Redis and Rate Limiting

Redis is used for distributed API rate limiting.

Rate-limit keys are scoped using identifiers such as:

```text
rate_limit:{scope}:user:{user_id}
```

or, for anonymous requests:

```text
rate_limit:{scope}:ip:{client_ip}
```

This prevents one authenticated user from consuming another user's quota.

Rate limiting is applied to selected authentication and application endpoints and returns controlled `429 Too Many Requests` responses when limits are exceeded.

---

## Secure Document Uploads

Document uploads are validated before RAG ingestion.

Protections include:

- allowed file extensions,
- safe filename validation,
- filename length limits,
- path traversal prevention,
- control-character rejection,
- content validation,
- UTF-8 validation for text documents,
- DOCX ZIP validation,
- protection against suspicious archive contents,
- ZIP compression-ratio limits,
- encrypted archive rejection,
- symbolic link rejection,
- randomized stored filenames using UUIDs,
- rollback when RAG ingestion fails,
- user-scoped RAG persistence.

Two users can upload documents with the same original filename without overwriting each other's files.

---

## API Security

The API contains multiple hardening layers.

### Trusted hosts

Requests with invalid `Host` headers are rejected.

### CORS

Only explicitly configured origins are allowed.

Wildcard origins are not accepted.

### Security headers

Production responses include headers such as:

```text
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
Permissions-Policy
Content-Security-Policy
```

### Request validation

Pydantic schemas validate API input.

Validation failures return standardized responses such as:

```json
{
  "success": false,
  "message": "Request validation failed.",
  "error_code": "request_validation_failed"
}
```

### Authentication failures

Protected resources return standardized responses:

```json
{
  "success": false,
  "message": "Authentication required.",
  "error_code": "authentication_required"
}
```

---

## LangSmith and Evaluation

LangSmith is used for tracing and observing LangGraph execution.

The project also contains a dedicated evaluation layer:

```text
evals/
├── data/
├── dataset.py
├── debug_reranker.py
├── evaluators.py
├── langsmith_client.py
├── regression.py
├── run_experiment.py
├── schemas.py
├── seed_corpus.py
├── simulate_regression_failure.py
├── sync_dataset.py
├── target.py
└── tracing.py
```

The evaluation system supports:

- LangSmith datasets,
- RAG evaluation cases,
- custom evaluators,
- automated experiment execution,
- regression detection,
- regression quality gates.

Regression evaluation is integrated into GitHub Actions and can block Pull Requests when quality requirements are not satisfied.

---

## Testing

The project contains extensive unit and integration tests.

```text
tests/
├── integration/
└── unit/
```

Tests cover areas including:

- LangGraph agent construction,
- graph nodes,
- routing,
- tools,
- streaming,
- RAG,
- query rewriting,
- reranking,
- configuration,
- database persistence,
- PostgreSQL integration,
- LangGraph checkpoints,
- conversations,
- authentication,
- session handling,
- rate limiting,
- API schemas,
- security headers,
- TrustedHost behavior,
- CORS,
- document upload security,
- multi-user isolation,
- cross-user access attempts,
- Human-in-the-Loop,
- evaluation pipeline,
- regression evaluation.

Run the complete test suite with:

```bash
python -m pytest -v
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

### Pull Request Quality Gates

Pull Requests run:

```text
Python tests
PostgreSQL integration tests
Docker build
Dependency security audit
Bandit static security analysis
Gitleaks secret scan
Trivy container vulnerability scan
Regression Quality Gate
```

Required checks must pass before merging into `main`.

---

## Security Tooling

### pip-audit

Production Python dependencies are scanned for known vulnerabilities.

### Bandit

Python source code is scanned for security issues using static analysis.

### Gitleaks

Repository history is scanned for accidentally committed credentials and secrets.

### Trivy

The final production Docker image is scanned for:

```text
HIGH
CRITICAL
```

known vulnerabilities with available fixes.

The CI pipeline fails when qualifying vulnerabilities are found.

### Immutable GitHub Actions

External GitHub Actions are pinned to complete commit SHA values rather than mutable version tags.

Example:

```yaml
uses: actions/checkout@<full-commit-sha> # vX.Y.Z
```

This reduces supply-chain risk from mutable GitHub Action tags.

---

## Deployment Pipeline

After changes pass Pull Request checks and are merged:

```mermaid
flowchart LR
    PR[Pull Request]
    CI[CI + Security Gates]
    REG[Regression Quality Gate]
    MAIN[Merge to main]
    IMG[Docker Image Publish]
    AWS[AWS Deployment]
    HEALTH[Health Check]
    PROD[Production]

    PR --> CI
    PR --> REG

    CI --> MAIN
    REG --> MAIN

    MAIN --> IMG
    IMG --> AWS
    AWS --> HEALTH
    HEALTH --> PROD
```

---

## Production Deployment

RobBotGPT is deployed on AWS.

Production infrastructure includes:

```text
AWS EC2
AWS IAM / OIDC
AWS Systems Manager
Docker
Docker Compose
PostgreSQL + pgvector
Redis
Nginx
Let's Encrypt
HTTPS
GitHub Actions
Docker Hub
```

The application is exposed through Nginx.

The FastAPI container listens internally on port:

```text
8080
```

---

### AWS Authentication

GitHub Actions uses **OpenID Connect (OIDC)** to assume an AWS IAM role.

GitHub Actions authenticates to AWS through OIDC instead of using long-lived AWS access keys for deployment.

---

### Deployment with AWS Systems Manager

Deployment commands are executed through **AWS Systems Manager (SSM)**.

The deployment workflow:

1. resolves the immutable image SHA,
2. validates deployment configuration,
3. updates Nginx configuration,
4. pulls the required Docker image,
5. starts the new application container,
6. performs health checks,
7. verifies the deployed image.

---

### Automatic Rollback

If the new application version fails its health check:

```text
new deployment
      ↓
health check fails
      ↓
previous Docker image restored
      ↓
application restarted
      ↓
rollback health check
```

The deployment fails if both the new version and rollback fail.

---

## Docker

The application uses a multi-stage Docker build based on Python 3.11.

The production runtime:

- runs as a non-root user,
- minimizes build dependencies,
- uses a dedicated virtual environment,
- drops Linux capabilities,
- prevents privilege escalation,
- supports read-only container configuration,
- exposes only required runtime resources.

The final image is scanned with Trivy in CI.

---

## Local Development

### 1. Clone the repository

```bash
git clone https://github.com/RobertStec/RobBotGPT.git
cd RobBotGPT
```

---

### 2. Create a Python environment

Using Conda:

```bash
conda create -n robbotgpt python=3.11 -y
conda activate robbotgpt
```

---

### 3. Install dependencies

Application dependencies:

```bash
pip install -r requirements.txt
```

Development and testing:

```bash
pip install -r requirements-dev.txt
```

---

### 4. Configure environment variables

Copy:

```bash
cp .env.example .env
```

Configure the required variables.

Never commit the real `.env` file.

---

### 5. Start infrastructure

```bash
docker compose up -d db redis
```

---

### 6. Run database migrations

```bash
alembic upgrade head
```

---

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

## Docker Compose

The complete local stack can be started with:

```bash
docker compose up --build
```

Stop it with:

```bash
docker compose down
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
OPENAI_MODEL=gpt-5.6-sol
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

### Redis

Example:

```env
REDIS_URL=redis://localhost:6379/0
```

Never store real production credentials in the repository.

---

## Database Migrations

Database schema changes are managed through Alembic.

Apply migrations:

```bash
alembic upgrade head
```

Current migrations include support for:

- initial database schema,
- users,
- authentication sessions,
- multi-user ownership,
- ownership constraints,
- pgvector-based RAG storage.

---

## Project Structure

```text
RobBotGPT/
│
├── api/
│   ├── dependencies/
│   │   └── auth.py
│   │
│   ├── routers/
│   │   ├── auth.py
│   │   ├── chat.py
│   │   ├── conversations.py
│   │   └── documents.py
│   │
│   └── schemas/
│       ├── auth.py
│       ├── chat.py
│       └── common.py
│
├── core/
│   ├── config.py
│   ├── environment.py
│   ├── exceptions.py
│   ├── lifecycle.py
│   ├── logging.py
│   ├── rate_limit.py
│   ├── security.py
│   └── security_headers.py
│
├── services/
│   ├── auth_service.py
│   ├── conversation_service.py
│   ├── document_service.py
│   └── streaming.py
│
├── evals/
│   ├── data/
│   ├── dataset.py
│   ├── evaluators.py
│   ├── langsmith_client.py
│   ├── regression.py
│   ├── run_experiment.py
│   ├── schemas.py
│   ├── seed_corpus.py
│   ├── sync_dataset.py
│   ├── target.py
│   └── tracing.py
│
├── tests/
│   ├── integration/
│   ├── unit/
│   └── conftest.py
│
├── alembic/
│   └── versions/
│
├── deploy/
│   └── nginx/
│       └── robbotgpt.conf
│
├── docker/
│   └── entrypoint.sh
│
├── docs/
│   └── SMOKE_TESTS.md
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── index.html
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
├── docker-compose.prod.yml
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

Generated or local-only artifacts such as `.env`, coverage reports, uploaded documents and local caches are intentionally excluded from this structure.

---

## Reliability and Security

RobBotGPT includes multiple reliability and security mechanisms:

- user authentication,
- password hashing,
- persistent sessions,
- secure production cookies,
- multi-user data isolation,
- user-scoped RAG,
- user-scoped LangGraph state,
- Redis-based rate limiting,
- safe document upload validation,
- UUID-based stored filenames,
- request schema validation,
- standardized API errors,
- TrustedHost protection,
- strict CORS configuration,
- security response headers,
- centralized logging,
- PostgreSQL health checks,
- application health checks,
- non-root Docker runtime,
- dropped Linux capabilities,
- `no-new-privileges`,
- dependency security auditing,
- Bandit static analysis,
- Gitleaks secret scanning,
- Trivy Docker vulnerability scanning,
- immutable GitHub Action SHAs,
- Pull Request quality gates,
- automated regression evaluation,
- automatic production rollback,
- RAG fallback instead of unsupported answers,
- Human-in-the-Loop for sensitive operations,
- safe calculator implementation without `eval()`.

---

## Production Security Verification

After deployment, production smoke tests verify:

```text
/health                       → 200
database connection           → healthy
main application              → 200

allowed CORS origin           → accepted
untrusted CORS origin         → rejected

trusted Host                  → accepted
invalid Host                  → rejected

invalid request payload       → 422
unauthenticated request       → 401
invalid session cookie        → 401
```

Security headers are also verified in the production environment.

---

## Example Use Cases

### General conversation

```text
Explain how LangGraph differs from a traditional LangChain agent.
```

---

### Document RAG

Upload a document and ask:

```text
Answer based on the uploaded document:
What is the main color of the Atlas project interface?
```

RobBotGPT retrieves relevant chunks and provides the answer together with document sources.

---

### Missing information

If requested information is not present in uploaded documents, the application returns a controlled fallback instead of inventing an answer.

---

### Long-term memory

```text
Remember that my project code is KAPPA-7319.
```

Later:

```text
What project code did I give you?
```

---

### Tool usage

```text
Calculate 125 * 48 / 6.
```

or:

```text
What is the current weather in Warsaw?
```

---

### Human-in-the-Loop

Selected actions can pause graph execution and require explicit user approval before continuing.

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

### Persistence

- PostgreSQL
- pgvector
- LangGraph PostgreSQL checkpoints
- Redis

### Infrastructure

- Docker
- Docker Compose
- GitHub Actions
- Docker Hub
- AWS EC2
- AWS IAM / OIDC
- AWS Systems Manager
- Nginx
- Let's Encrypt

### Security

- Bandit
- pip-audit
- Gitleaks
- Trivy
- TrustedHost
- CORS hardening
- secure session cookies
- Redis rate limiting

### Testing

- pytest
- unit tests
- API integration tests
- PostgreSQL integration tests
- multi-user isolation tests
- security tests
- regression evaluation
- LangSmith datasets and evaluators

---

## Future Improvements

RobBotGPT v1.0 already contains the core production architecture planned for the project.

Possible future extensions include:

- Infrastructure as Code,
- Terraform-based AWS provisioning,
- Kubernetes deployment,
- horizontal application scaling,
- managed PostgreSQL and Redis,
- distributed background task processing,
- additional agent tools,
- richer metrics and dashboards,
- OpenTelemetry integration,
- automated release versioning,
- image signing and provenance,
- additional RAG evaluation datasets,
- load and performance testing.

These are optional extensions rather than requirements for the current production version.

---

## Project Status

```text
RobBotGPT v1.0
Status: Production-deployed portfolio project
```

The application has passed:

- unit tests,
- integration tests,
- PostgreSQL integration tests,
- multi-user isolation tests,
- security regression tests,
- LangSmith regression evaluation,
- dependency audits,
- static security analysis,
- secret scanning,
- Docker vulnerability scanning,
- production deployment smoke tests.

---

## License

This project includes a `LICENSE` file.

See the license file for details.

---

## Author

**Robert Stec**

RobBotGPT was developed as a portfolio project focused on practical AI engineering, LangGraph agent architecture, Retrieval-Augmented Generation, multi-user security, evaluation, testing, CI/CD and production deployment.