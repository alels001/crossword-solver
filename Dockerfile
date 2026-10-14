# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[openai,anthropic,api,mcp,ui]"

FROM python:3.12-slim
# libgl/glib are needed by OpenCV even in headless builds on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY --chown=app:app data ./data
COPY --chown=app:app apps ./apps
ENV CWS_KNOWLEDGE_BASE_PATH=/app/data/knowledge/clues_de.csv \
    CWS_BENCHMARK_DIR=/app/data/benchmark \
    CWS_CACHE_DIR=/app/.cache \
    CWS_LOG_JSON=true \
    PYTHONUNBUFFERED=1
RUN mkdir -p /app/.cache && chown app:app /app/.cache
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "crossword_solver.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
