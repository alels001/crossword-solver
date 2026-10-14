"""HTTP API (FastAPI).

Run: ``uvicorn crossword_solver.serving.api:app`` or ``crossword serve``.
OpenAPI docs are served at ``/docs``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..config import get_settings
from ..llm import BudgetExceededError
from ..models import Candidate, PuzzleLayout
from ..observability import METRICS, request_id, setup_logging
from ..pipeline import CrosswordSolver

log = logging.getLogger(__name__)
settings = get_settings()
setup_logging(settings.log_level, settings.log_json)

app = FastAPI(
    title="Crossword Solver API",
    version=__version__,
    description="Solve German crosswords from a photo or a structured layout: vision → OCR → RAG → LLM → CSP → agent.",
)

_solver: CrosswordSolver | None = None
_solver_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict[str, Any]] = {}
_ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}


def get_solver() -> CrosswordSolver:
    global _solver
    with _solver_lock:
        if _solver is None:
            _solver = CrosswordSolver(settings)
        return _solver


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.middleware("http")
async def _observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id.set(rid)
    t0 = time.perf_counter()
    try:
        response: Response = await call_next(request)
    finally:
        request_id.reset(token)
    elapsed = time.perf_counter() - t0
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    METRICS.inc("cws_http_requests_total", path=path, status=str(response.status_code))
    METRICS.inc("cws_http_request_seconds_sum", elapsed, path=path)
    response.headers["x-request-id"] = rid
    return response


# ---- schemas --------------------------------------------------------------------
class CandidateRequest(BaseModel):
    clue: str = Field(min_length=1, max_length=300, examples=["sizilianischer Vulkan"])
    length: int = Field(ge=1, le=30, examples=[5])
    pattern: str | None = Field(default=None, examples=["A__NA"], description="'_' for unknown letters")


class CandidateResponse(BaseModel):
    candidates: list[Candidate]
    usage: dict[str, float | int]


class LayoutRequest(BaseModel):
    layout: PuzzleLayout
    known_answers: dict[str, str] = Field(default_factory=dict)
    use_agent: bool = True


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


# ---- endpoints ------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, Any]:
    s = get_solver()
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": settings.llm_provider,
        "kb_records": len(s.store),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return METRICS.render()


@app.post("/v1/candidates", response_model=CandidateResponse, dependencies=[Depends(require_api_key)])
def candidates(req: CandidateRequest) -> CandidateResponse:
    from ..llm import UsageTracker

    tracker = UsageTracker(settings.max_cost_usd_per_puzzle)
    try:
        cands = get_solver().suggest(req.clue, req.length, req.pattern, tracker=tracker)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    METRICS.inc("cws_llm_cost_usd_total", tracker.cost_usd, endpoint="candidates")
    return CandidateResponse(candidates=cands, usage=tracker.snapshot())


@app.post("/v1/solve/layout", dependencies=[Depends(require_api_key)])
def solve_layout(req: LayoutRequest) -> dict[str, Any]:
    try:
        report = get_solver().solve_layout(
            req.layout, known_answers=req.known_answers, use_agent=req.use_agent
        )
    except BudgetExceededError as e:
        raise HTTPException(status_code=402, detail=str(e)) from e
    METRICS.inc("cws_llm_cost_usd_total", report.solution.usage.cost_usd, endpoint="solve_layout")
    return report.to_dict()


async def _read_upload(file: UploadFile) -> bytes:
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"unsupported content type {file.content_type}")
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"image larger than {settings.max_upload_mb} MB")
    return data


def _run_job(job_id: str, data: bytes, use_agent: bool) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    request_id.set(f"job-{job_id[:8]}")  # correlate background log lines with the job
    try:
        report = get_solver().solve_image(
            data, use_agent=use_agent, on_progress=lambda m: job.update(progress=m)
        )
        job.update(status="done", result=report.to_dict())
        METRICS.inc("cws_jobs_total", status="done")
    except Exception as e:  # surfaced to the client, logged with stack trace
        log.exception("job_failed", extra={"job_id": job_id})
        job.update(status="failed", error=str(e))
        METRICS.inc("cws_jobs_total", status="failed")


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(require_api_key)])
async def create_job(file: Annotated[UploadFile, File()], use_agent: bool = True) -> dict[str, str]:
    """Upload a crossword photo; solving runs in the background (vision + OCR + LLM can take a minute)."""
    data = await _read_upload(file)
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued", "created": time.time(), "progress": "queued"}
    _executor.submit(_run_job, job_id, data, use_agent)
    return {"job_id": job_id, "status_url": f"/v1/jobs/{job_id}"}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {"job_id": job_id, **job}


@app.post("/v1/chat", dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest) -> dict[str, str]:
    reply = get_solver().chat(req.message, [t.model_dump() for t in req.history])
    return {"reply": reply}
