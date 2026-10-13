# =========================================================
# Stage 1 — dependencies builder
# =========================================================

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VIRTUAL_ENV}"

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /build

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt


# =========================================================
# Stage 2 — runtime
# =========================================================

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY . .

RUN sed -i 's/\r$//' /app/docker/entrypoint.sh \
    && chmod +x /app/docker/entrypoint.sh

RUN mkdir -p \
        /app/uploads \
        /app/data \
    && addgroup --system robbotgpt \
    && adduser --system \
        --ingroup robbotgpt \
        robbotgpt \
    && chown -R robbotgpt:robbotgpt /app

USER robbotgpt

EXPOSE 8080

ENTRYPOINT ["/app/docker/entrypoint.sh"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]