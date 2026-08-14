from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from paperless_clerk import __version__
from paperless_clerk.clients.openai_compatible import ModelError, OpenAICompatibleClient
from paperless_clerk.clients.paperless import PaperlessClient, PaperlessError
from paperless_clerk.config import SettingsManager, data_directory
from paperless_clerk.db import Database
from paperless_clerk.processing import JobManager, ProcessingError, resolve_conflict
from paperless_clerk.schemas import EnqueueRequest, ResolveConflictRequest, SettingsPatch

log = logging.getLogger(__name__)
STATIC_DIRECTORY = Path(__file__).parent / "static"


def _configure_application_logging(
    level_name: str, *, stream: TextIO | None = None
) -> logging.Logger:
    """Give Clerk its own Docker-visible handler instead of relying on Uvicorn's root setup."""
    package_logger = logging.getLogger("paperless_clerk")
    package_logger.handlers.clear()
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(getattr(logging, level_name))
    package_logger.propagate = False
    package_logger.disabled = False
    return package_logger


def _timestamp(value: float | None) -> str | None:
    return datetime.fromtimestamp(value, UTC).isoformat() if value else None


def _serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    result = dict(job)
    # Hide a legacy, pre-release column if an early development database is
    # opened after upgrading.
    result.pop("force", None)
    for internal in ("source_hash", "ocr_fingerprint", "worker_id"):
        result.pop(internal, None)
    for field in (
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
        "next_run_at",
        "lease_until",
    ):
        result[field] = _timestamp(result.get(field))
    total = int(result.get("progress_total") or 0)
    current = int(result.get("progress_current") or 0)
    result["progress_percent"] = round((current / total) * 100) if total else 0
    if "events" in result:
        for event in result["events"]:
            event["created_at"] = _timestamp(event.get("created_at"))
    return result


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    for field in ("created_at", "resolved_at", "processed_at"):
        if field in result:
            result[field] = _timestamp(result.get(field))
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    directory = data_directory()
    database = Database(directory / "clerk.db")
    database.initialize()
    settings_manager = SettingsManager(database)
    settings = settings_manager.get()
    _configure_application_logging(settings.log_level)
    manager = JobManager(database, settings_manager)
    app.state.database = database
    app.state.settings_manager = settings_manager
    app.state.job_manager = manager
    await manager.start()
    log.info(
        "Paperless Clerk %s started with data directory %s and log level %s",
        __version__,
        directory,
        settings.log_level,
    )
    try:
        yield
    finally:
        await manager.stop()
        log.info("Paperless Clerk stopped")


app = FastAPI(
    title="Paperless Clerk",
    description="Local-AI document intelligence for Paperless-ngx",
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


@app.middleware("http")
async def private_api_cache_control(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    if request.url.path.startswith("/api/") and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-store"
    return response


def _database(request: Request) -> Database:
    return request.app.state.database


def _settings_manager(request: Request) -> SettingsManager:
    return request.app.state.settings_manager


@app.exception_handler(ProcessingError)
async def processing_error_handler(_: Request, exc: ProcessingError) -> JSONResponse:
    if exc.code == "conflict_not_found":
        code = status.HTTP_404_NOT_FOUND
    elif exc.code in {"paperless_error", "model_error"}:
        code = status.HTTP_502_BAD_GATEWAY
    else:
        code = status.HTTP_409_CONFLICT
    return JSONResponse(status_code=code, content={"error": exc.code, "detail": str(exc)})


@app.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    settings = _settings_manager(request).get()
    return {
        "status": "ok",
        "version": __version__,
        "configured": {
            "paperless": bool(settings.paperless_token.get_secret_value()),
            "ocr": bool(settings.openai_base_url and settings.ocr_model),
            "metadata": bool(settings.openai_base_url and settings.metadata_model),
        },
    }


@app.get("/api/dashboard")
async def dashboard(request: Request) -> dict[str, Any]:
    database = _database(request)
    settings = _settings_manager(request).get()
    return {
        "counts": database.dashboard_counts(),
        "jobs": [_serialize_job(job) for job in database.list_jobs(limit=8)],
        "conflicts": [_serialize_record(item) for item in database.list_conflicts(limit=5)],
        "decisions": [_serialize_record(item) for item in database.list_decisions(limit=5)],
        "paperless_url": settings.paperless_url.removesuffix("/api"),
        "automation_enabled": settings.automation_enabled,
        "automation_tag": settings.automation_tag,
    }


@app.get("/api/jobs")
async def list_jobs(
    request: Request,
    job_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    return [
        _serialize_job(job) for job in _database(request).list_jobs(status=job_status, limit=limit)
    ]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict[str, Any]:
    job = _database(request).get_job(job_id, include_events=True)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _serialize_job(job)


@app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_jobs(payload: EnqueueRequest, request: Request) -> dict[str, Any]:
    database = _database(request)
    settings = _settings_manager(request).get()
    conflicted = [
        document_id
        for document_id in payload.document_ids
        if database.has_open_conflict(document_id)
    ]
    if conflicted:
        raise HTTPException(
            status_code=409,
            detail=(
                "Resolve the open OCR conflict before reprocessing document(s): "
                + ", ".join(str(document_id) for document_id in conflicted)
            ),
        )
    jobs = []
    for document_id in payload.document_ids:
        job, created = database.enqueue_job(
            document_id,
            payload.mode,
            settings.job_max_attempts,
        )
        jobs.append({"created": created, "job": _serialize_job(job)})
    request.app.state.job_manager.wake()
    return {"jobs": jobs}


@app.post("/api/jobs/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_job(job_id: str, request: Request) -> dict[str, Any]:
    job = _database(request).retry_job(job_id)
    if not job:
        raise HTTPException(status_code=409, detail="Job is not retryable or another job is active")
    request.app.state.job_manager.wake()
    return _serialize_job(job)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request) -> dict[str, bool]:
    if not _database(request).cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Only queued jobs can be cancelled")
    return {"cancelled": True}


@app.get("/api/conflicts")
async def list_conflicts(
    request: Request,
    conflict_status: str = Query(default="open", alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    return [
        _serialize_record(item)
        for item in _database(request).list_conflicts(conflict_status, limit)
    ]


@app.get("/api/conflicts/{conflict_id}")
async def get_conflict(conflict_id: str, request: Request) -> dict[str, Any]:
    conflict = _database(request).get_conflict(conflict_id)
    if not conflict:
        raise HTTPException(status_code=404, detail="Conflict not found")
    return _serialize_record(conflict)


@app.post("/api/conflicts/{conflict_id}/resolve", status_code=status.HTTP_202_ACCEPTED)
async def resolve_ocr_conflict(
    conflict_id: str, payload: ResolveConflictRequest, request: Request
) -> dict[str, Any]:
    result = await resolve_conflict(
        database=_database(request),
        settings=_settings_manager(request).get(),
        conflict_id=conflict_id,
        resolution=payload.resolution,
    )
    request.app.state.job_manager.wake()
    result["job"] = _serialize_job(result["job"])
    return result


@app.get("/api/decisions")
async def list_decisions(
    request: Request, limit: int = Query(default=100, ge=1, le=500)
) -> list[dict[str, Any]]:
    return [_serialize_record(item) for item in _database(request).list_decisions(limit)]


@app.get("/api/decisions/{decision_id}")
async def get_decision(decision_id: str, request: Request) -> dict[str, Any]:
    decision = _database(request).get_decision(decision_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    return _serialize_record(decision)


@app.get("/api/interventions")
async def interventions(request: Request) -> dict[str, Any]:
    database = _database(request)
    failed = database.list_jobs(status="failed", limit=100)
    review = database.list_jobs(status="needs_review", limit=100)
    return {
        "conflicts": [_serialize_record(item) for item in database.list_conflicts("open", 100)],
        "failed_jobs": [_serialize_job(item) for item in failed],
        "review_jobs": [_serialize_job(item) for item in review],
    }


@app.get("/api/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    return _settings_manager(request).get().public_dict()


@app.patch("/api/settings")
async def update_settings(payload: SettingsPatch, request: Request) -> dict[str, Any]:
    before = _settings_manager(request).get()
    try:
        updated = _settings_manager(request).update(payload.values)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    restart_fields = {"job_workers", "log_level"}
    restart_required = sorted(
        field
        for field in restart_fields & payload.values.keys()
        if getattr(before, field) != getattr(updated, field)
    )
    request.app.state.job_manager.settings_changed()
    return {"settings": updated.public_dict(), "restart_required": restart_required}


@app.post("/api/settings/test/{target}")
async def test_settings(target: str, request: Request) -> dict[str, Any]:
    settings = _settings_manager(request).get()
    if target == "paperless":
        client = PaperlessClient(settings)
    elif target in {"ocr", "metadata"}:
        client = OpenAICompatibleClient(settings, target)
    else:
        raise HTTPException(status_code=404, detail="Unknown connection target")
    try:
        return await client.test_connection()
    except (PaperlessError, ModelError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        await client.close()


@app.get("/api/documents/{document_id}/thumbnail")
async def document_thumbnail(document_id: int, request: Request) -> Response:
    client = PaperlessClient(_settings_manager(request).get())
    try:
        content, content_type = await client.thumbnail(document_id)
        return Response(
            content=content,
            media_type=content_type,
            headers={"Cache-Control": "private, max-age=300"},
        )
    except PaperlessError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        await client.close()


app.mount("/assets", StaticFiles(directory=STATIC_DIRECTORY), name="assets")


@app.get("/{path:path}", include_in_schema=False)
async def single_page_app(path: str) -> FileResponse:
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    return FileResponse(STATIC_DIRECTORY / "index.html", headers={"Cache-Control": "no-cache"})


def run() -> None:
    uvicorn.run(
        "paperless_clerk.main:app",
        host=os.environ.get("CLERK_HOST", "0.0.0.0"),
        port=int(os.environ.get("CLERK_PORT", "8080")),
        proxy_headers=True,
    )
