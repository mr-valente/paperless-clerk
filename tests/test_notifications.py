import asyncio
import json
from pathlib import Path

import httpx
import pytest

import paperless_clerk.processing as processing
from paperless_clerk.clients.ntfy import NotificationError, NtfyClient
from paperless_clerk.config import Settings, SettingsManager
from paperless_clerk.db import Database
from paperless_clerk.processing import JobManager, ProcessingError


@pytest.mark.asyncio
async def test_ntfy_client_uses_json_api_and_optional_bearer_token() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "message-id", "event": "message"})

    client = NtfyClient(Settings(ntfy_topic="clerk_alerts", ntfy_token="secret-notification-token"))
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer secret-notification-token",
        },
    )

    await client.publish(
        title="Paperless Clerk: processing failed",
        message="Invoice (document #42)\nOCR failed",
        priority=4,
        tags=("x", "page_facing_up"),
    )
    await client.close()

    assert len(requests) == 1
    assert str(requests[0].url) == "https://ntfy.sh"
    assert requests[0].headers["authorization"] == "Bearer secret-notification-token"
    assert json.loads(requests[0].content) == {
        "topic": "clerk_alerts",
        "title": "Paperless Clerk: processing failed",
        "message": "Invoice (document #42)\nOCR failed",
        "priority": 4,
        "tags": ["x", "page_facing_up"],
    }


@pytest.mark.asyncio
async def test_notification_delivery_failure_is_recorded_without_escaping_job_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingNtfyClient:
        def __init__(self, settings: Settings):
            assert settings.notifications_enabled is True

        async def publish(self, **_: object) -> None:
            raise NotificationError("topic denied")

        async def close(self) -> None:
            pass

    database = Database(tmp_path / "clerk.db")
    database.initialize()
    settings_manager = SettingsManager(database)
    settings_manager.update({"notifications_enabled": True, "ntfy_topic": "clerk-private-alerts"})
    job, _ = database.enqueue_job(42, "full", 3, document_title="Veterinary invoice")
    manager = JobManager(database, settings_manager)
    monkeypatch.setattr(processing, "NtfyClient", FailingNtfyClient)

    await manager._notify_job_issue(  # noqa: SLF001
        job, kind="ocr_conflict", message="OCR versions differ"
    )

    detail = database.get_job(job["id"], include_events=True)
    assert detail is not None
    notification = next(
        event for event in detail["events"] if event["event_type"] == "notification_failed"
    )
    assert notification["data"] == {"kind": "ocr_conflict"}
    assert "topic denied" in notification["message"]


@pytest.mark.asyncio
async def test_job_issue_notification_contains_bounded_document_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published: list[dict] = []

    class CapturingNtfyClient:
        def __init__(self, settings: Settings):
            assert settings.ntfy_topic == "clerk-private-alerts"

        async def publish(self, **values: object) -> None:
            published.append(values)

        async def close(self) -> None:
            pass

    database = Database(tmp_path / "clerk.db")
    database.initialize()
    settings_manager = SettingsManager(database)
    settings_manager.update({"notifications_enabled": True, "ntfy_topic": "clerk-private-alerts"})
    job, _ = database.enqueue_job(43, "full", 3, document_title="Veterinary invoice")
    manager = JobManager(database, settings_manager)
    monkeypatch.setattr(processing, "NtfyClient", CapturingNtfyClient)

    await manager._notify_job_issue(job, kind="failed", message="Model timed out")  # noqa: SLF001

    assert published == [
        {
            "title": "Paperless Clerk: processing failed",
            "message": "Veterinary invoice (document #43)\nModel timed out",
            "priority": 4,
            "tags": ("x", "page_facing_up"),
        }
    ]
    detail = database.get_job(job["id"], include_events=True)
    assert detail is not None
    assert "notification_sent" in {event["event_type"] for event in detail["events"]}


@pytest.mark.asyncio
async def test_worker_sends_notification_when_processing_reaches_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delivered = asyncio.Event()
    published: list[dict] = []

    class CapturingNtfyClient:
        def __init__(self, _: Settings):
            pass

        async def publish(self, **values: object) -> None:
            published.append(values)
            delivered.set()

        async def close(self) -> None:
            pass

    class FailingProcessor:
        def __init__(self, *_: object):
            pass

        async def process(self, _: dict) -> None:
            raise ProcessingError("model_error", "OCR endpoint rejected the page")

    database = Database(tmp_path / "clerk.db")
    database.initialize()
    settings_manager = SettingsManager(database)
    settings_manager.update({"notifications_enabled": True, "ntfy_topic": "clerk-private-alerts"})
    job, _ = database.enqueue_job(44, "full", 3, document_title="Failed invoice")
    manager = JobManager(database, settings_manager)
    monkeypatch.setattr(processing, "NtfyClient", CapturingNtfyClient)
    monkeypatch.setattr(processing, "DocumentProcessor", FailingProcessor)

    worker = asyncio.create_task(manager._worker(0))  # noqa: SLF001
    try:
        await asyncio.wait_for(delivered.wait(), timeout=2)
    finally:
        manager._stop.set()  # noqa: SLF001
        manager.wake()
        await asyncio.wait_for(worker, timeout=2)

    detail = database.get_job(job["id"], include_events=True)
    assert detail is not None
    assert detail["status"] == "failed"
    assert published[0]["title"] == "Paperless Clerk: processing failed"
    assert "OCR endpoint rejected the page" in published[0]["message"]
