# Operations

## Deploy

```bash
cp .env.example .env     # provider, keys, budgets, optional CWS_API_KEY
docker compose up -d --build
curl localhost:8000/healthz
```

The image runs as a non-root user, has a health check, logs JSON to stdout and keeps the LLM response cache on a
named volume. One Uvicorn process serves requests (LLM calls are I/O-bound and run in threads); photo solves run in a
background thread pool (`/v1/jobs`). The job store is in memory on purpose — to scale to several workers or replicas,
move it to Redis or a database first, otherwise a status request can land on a process that never saw the job.

## Configuration reference (most used)

| Variable | Default | Meaning |
|---|---|---|
| `CWS_LLM_PROVIDER` | `fake` | `openai`, `anthropic` or `fake` (offline) |
| `CWS_OPENAI_MODEL` / `CWS_OPENAI_VISION_MODEL` | `gpt-4.1-mini` / `gpt-4.1` | Text and vision models |
| `CWS_ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Model for the Anthropic provider |
| `CWS_RETRIEVER` | `tfidf` | `tfidf`, `embedding`, `hybrid` |
| `CWS_MAX_COST_USD_PER_PUZZLE` | `0.50` | Hard budget per solve or request |
| `CWS_AGENT_MAX_STEPS` | `25` | Tool-call limit for the agent |
| `CWS_ENABLE_WEB_DICTIONARY` | `false` | Scrape wort-suchen.de (cached, rate-limited) |
| `CWS_API_KEY` | – | Require `X-API-Key` on API calls |
| `CWS_LOG_JSON` | `false` | JSON logs for log aggregation |

## Monitoring

* **Logs**: every LLM call logs model, latency, input/output tokens, cache hit and number of tool calls; every agent
  run logs steps and placements; every line carries the request id (`x-request-id` header is echoed).
* **Metrics** (`GET /metrics`): request counts and latency sums per route and status, LLM cost per endpoint, job
  outcomes, uptime.
* **Per response**: `solution.usage` (calls, tokens, cost, cache hits) and `timings_ms` per stage.

Suggested alerts: error rate on `/v1/jobs`, p95 of `cws_http_request_seconds_sum / cws_http_requests_total`,
daily `cws_llm_cost_usd_total`.

## Cost control

1. Response cache — identical prompts are never billed twice (also makes re-runs of the eval cheap).
2. Budget per puzzle — `TrackedLLM` raises `BudgetExceededError` before a call once the budget is spent; the pipeline
   catches it and returns the partial solution with `stats.budget_exhausted = 1`.
3. Cheap model for text, stronger model only for vision OCR; the agent only runs on entries the solver left open.

## Runbook

| Symptom | Check | Action |
|---|---|---|
| Many open entries | `stats.entries_filled`, agent transcript | Raise `CWS_AGENT_MAX_STEPS`, enable web dictionary, extend the KB |
| Wrong grid structure from photos | `debug["structure"]` image from `analyse_image` | Adjust `CWS_GRID_ROWS/COLS`, `CWS_MASK_REGION` |
| `stats.budget_exhausted = 1` in a solution | `solution.usage.cost_usd` | Raise `CWS_MAX_COST_USD_PER_PUZZLE` or use a cheaper model |
| Provider outage | logs: `llm_call` errors | Switch `CWS_LLM_PROVIDER`; SDK retries (`CWS_MAX_RETRIES`) already back off |
