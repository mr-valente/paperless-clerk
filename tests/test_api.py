import io
import logging
from pathlib import Path

import httpx
import pytest

from paperless_clerk.config import SettingsManager
from paperless_clerk.db import Database
from paperless_clerk.main import _configure_application_logging, app


class FakeManager:
    def __init__(self):
        self.wakes = 0
        self.settings_changes = 0

    def wake(self) -> None:
        self.wakes += 1

    def settings_changed(self) -> None:
        self.settings_changes += 1


@pytest.mark.asyncio
async def test_production_lifespan_serves_health_ui_and_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLERK_DATA_DIR", str(tmp_path))
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://clerk.local") as client,
    ):
        health = await client.get("/api/health")
        page = await client.get("/")
        script = await client.get("/assets/app.js")
        styles = await client.get("/assets/styles.css")
        favicon = await client.get("/assets/favicon.svg")

    assert health.status_code == 200
    assert health.headers["cache-control"] == "no-store"
    assert page.status_code == 200
    assert "Paperless Clerk" in page.text
    assert script.status_code == 200
    assert "renderOverview" in script.text
    assert "ocr_profile_choices" in script.text
    assert "Keep original document version" in script.text
    assert "Prefer Clerk OCR after a trusted match" not in script.text
    assert "OCR review versions" in script.text
    assert "Paperless had no OCR baseline" in script.text
    assert "Both complete OCR versions" not in script.text
    assert "Either choice removes the conflict tag" not in script.text
    assert "View diagnostic log" in script.text
    assert "decisionDiagnosticLog" in script.text
    assert "toggle-decision-log" in script.text
    assert "Container log detail" in script.text
    assert "Enable ntfy notifications" in script.text
    assert "View decision" in script.text
    assert "progress-track indeterminate" in script.text
    assert "Every run has a durable paper trail" not in script.text
    assert "Classification you can account for" not in script.text
    assert "Nothing fails silently" not in script.text
    assert "Private by design" not in page.text
    assert '.replace(/\\bOcr\\b/g, "OCR")' in script.text
    assert favicon.status_code == 200
    assert favicon.headers["content-type"].startswith("image/svg+xml")
    assert styles.status_code == 200
    assert ".grid > .panel + .panel" in styles.text
    assert (tmp_path / "clerk.db").exists()


@pytest.mark.asyncio
async def test_health_settings_and_manual_enqueue_api(tmp_path: Path) -> None:
    database = Database(tmp_path / "clerk.db")
    database.initialize()
    app.state.database = database
    app.state.settings_manager = SettingsManager(database)
    app.state.job_manager = FakeManager()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://clerk.local") as client:
        health = await client.get("/api/health")
        settings = await client.get("/api/settings")
        changed = await client.patch(
            "/api/settings", json={"values": {"keep_original_version": False}}
        )
        queued = await client.post(
            "/api/jobs", json={"document_ids": [42, 42, 43], "mode": "metadata"}
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "paperless_token" not in settings.json()
    assert "openai_base_url" in settings.json()
    assert "openai_api_key" not in settings.json()
    assert "openai_api_key_configured" in settings.json()
    assert "ntfy_token" not in settings.json()
    assert "ntfy_token_configured" in settings.json()
    assert settings.json()["notifications_enabled"] is False
    assert settings.json()["ocr_profile"] == "generic"
    assert settings.json()["keep_original_version"] is True
    assert changed.status_code == 200
    assert changed.json()["settings"]["keep_original_version"] is False
    assert changed.json()["restart_required"] == []
    # The vLLM profiles are held back, so the UI offers only the working two.
    assert [choice["key"] for choice in settings.json()["ocr_profile_choices"]] == [
        "generic",
        "deepseek_ocr_llamacpp",
    ]
    assert settings.json()["log_level"] == "INFO"
    assert "ocr_base_url" not in settings.json()
    assert "metadata_base_url" not in settings.json()
    assert [item["job"]["document_id"] for item in queued.json()["jobs"]] == [42, 43]
    assert app.state.job_manager.wakes == 1
    assert app.state.job_manager.settings_changes == 1


def test_clerk_logging_has_a_dedicated_stream_handler() -> None:
    package_logger = logging.getLogger("paperless_clerk")
    original_handlers = list(package_logger.handlers)
    original_level = package_logger.level
    original_propagate = package_logger.propagate
    original_disabled = package_logger.disabled
    stream = io.StringIO()
    try:
        configured = _configure_application_logging("INFO", stream=stream)
        logging.getLogger("paperless_clerk.processing").info(
            "Metadata decision recorded document=%s tags=%s", 9, 1
        )

        assert configured is package_logger
        assert configured.propagate is False
        assert "Metadata decision recorded document=9 tags=1" in stream.getvalue()
    finally:
        package_logger.handlers = original_handlers
        package_logger.setLevel(original_level)
        package_logger.propagate = original_propagate
        package_logger.disabled = original_disabled
