import io
import logging
from pathlib import Path

import httpx
import pytest

from paperless_clerk.config import SettingsManager
from paperless_clerk.db import Database
from paperless_clerk.main import (
    STATIC_ASSET_NAMES,
    STATIC_ASSET_VERSION,
    _configure_application_logging,
    _static_asset_version,
    app,
)


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
        script = await client.get(f"/assets/app.js?v={STATIC_ASSET_VERSION}")
        styles = await client.get(f"/assets/styles.css?v={STATIC_ASSET_VERSION}")
        favicon = await client.get(f"/assets/favicon.svg?v={STATIC_ASSET_VERSION}")
        unversioned_script = await client.get("/assets/app.js")

    assert health.status_code == 200
    assert health.headers["cache-control"] == "no-store"
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-cache"
    assert "Paperless Clerk" in page.text
    assert f"/assets/app.js?v={STATIC_ASSET_VERSION}" in page.text
    assert f"/assets/styles.css?v={STATIC_ASSET_VERSION}" in page.text
    assert f"/assets/favicon.svg?v={STATIC_ASSET_VERSION}" in page.text
    assert script.status_code == 200
    assert script.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert styles.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert favicon.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert unversioned_script.headers["cache-control"] == "no-cache"
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


@pytest.mark.parametrize("changed_name", STATIC_ASSET_NAMES)
def test_static_asset_version_changes_with_asset_content(tmp_path: Path, changed_name: str) -> None:
    for name in STATIC_ASSET_NAMES:
        (tmp_path / name).write_text(f"initial {name}")
    initial = _static_asset_version(tmp_path)

    (tmp_path / changed_name).write_text(f"changed {changed_name}")

    changed = _static_asset_version(tmp_path)
    assert len(initial) == 16
    assert initial != changed


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


@pytest.mark.asyncio
async def test_manual_enqueue_carries_a_per_job_ocr_retention_choice(tmp_path: Path) -> None:
    database = Database(tmp_path / "clerk.db")
    database.initialize()
    app.state.database = database
    app.state.settings_manager = SettingsManager(database)
    app.state.job_manager = FakeManager()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://clerk.local") as client:
        replacing = await client.post(
            "/api/jobs",
            json={"document_ids": [51], "mode": "ocr", "keep_original_version": False},
        )
        keeping = await client.post(
            "/api/jobs",
            json={"document_ids": [52], "mode": "full", "keep_original_version": True},
        )
        following = await client.post("/api/jobs", json={"document_ids": [53], "mode": "full"})
        detail = await client.get(f"/api/jobs/{replacing.json()['jobs'][0]['job']['id']}")

    # The API reports the stored SQLite integer as a bool, and None still means
    # the job follows whatever the setting says when it runs.
    assert replacing.json()["jobs"][0]["job"]["keep_original_version"] is False
    assert keeping.json()["jobs"][0]["job"]["keep_original_version"] is True
    assert following.json()["jobs"][0]["job"]["keep_original_version"] is None
    assert detail.json()["keep_original_version"] is False


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


@pytest.mark.asyncio
async def test_clear_jobs_removes_only_failed_and_cancelled_runs(tmp_path: Path) -> None:
    database = Database(tmp_path / "clerk.db")
    database.initialize()
    app.state.database = database
    app.state.settings_manager = SettingsManager(database)
    app.state.job_manager = FakeManager()
    failed, _ = database.enqueue_job(1, "full", 3)
    cancelled, _ = database.enqueue_job(2, "full", 3)
    completed, _ = database.enqueue_job(3, "full", 3)
    queued, _ = database.enqueue_job(4, "full", 3)
    database.fail_or_retry(failed["id"], "model_error", "boom", retryable=False)
    database.add_event(failed["id"], "info", "note", "an event to cascade")
    assert database.cancel_job(cancelled["id"])
    database.finish_job(completed["id"])
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://clerk.local") as client:
        cleared = await client.post("/api/jobs/clear", json={"statuses": ["failed", "cancelled"]})
        rejected = await client.post("/api/jobs/clear", json={"statuses": ["completed"]})
        remaining = await client.get("/api/jobs")

    assert cleared.status_code == 200
    assert cleared.json() == {"deleted": 2}
    assert rejected.status_code == 422
    assert sorted(job["document_id"] for job in remaining.json()) == [3, 4]
    assert database.get_job(failed["id"], include_events=True) is None
    assert database.get_job(queued["id"]) is not None


@pytest.mark.asyncio
async def test_vocabulary_review_merge_and_dismiss_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import paperless_clerk.main as main

    database = Database(tmp_path / "clerk.db")
    database.initialize()
    app.state.database = database
    app.state.settings_manager = SettingsManager(database)
    app.state.job_manager = FakeManager()
    suggestions = [
        {
            "key": "tag:3:4",
            "kind": "tag",
            "keep_id": 3,
            "keep_name": "Optical",
            "keep_usage": 2,
            "merge_id": 4,
            "merge_name": "Optometry",
            "merge_usage": 1,
            "reason": "Same subject",
            "source": "model",
        },
        {
            "key": "tag:1:2",
            "kind": "tag",
            "keep_id": 1,
            "keep_name": "Taxes",
            "keep_usage": 6,
            "merge_id": 2,
            "merge_name": "Income Taxes",
            "merge_usage": 1,
            "reason": "modifier",
            "source": "lexical",
        },
    ]
    merges: list[tuple] = []

    async def fake_review(settings, paperless, model):
        return {
            "reviewed_at": 1_700_000_000.0,
            "counts": {"tags": 4, "document_types": 1, "correspondents": 2},
            "model_status": "completed",
            "suggestions": suggestions,
        }

    async def fake_merge(paperless, kind, keep_id, merge_id):
        merges.append((kind, keep_id, merge_id))
        return {
            "kind": kind,
            "keep": {"id": keep_id, "name": "Optical"},
            "merged": {"id": merge_id, "name": "Optometry"},
            "documents": 2,
        }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(main, "review_vocabulary", fake_review)
    monkeypatch.setattr(main, "merge_entities", fake_merge)
    monkeypatch.setattr(main, "PaperlessClient", FakeClient)
    monkeypatch.setattr(main, "OpenAICompatibleClient", FakeClient)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://clerk.local") as client:
        empty = await client.get("/api/vocabulary")
        reviewed = await client.post("/api/vocabulary/review")
        dismissed = await client.post(
            "/api/vocabulary/dismiss", json={"kind": "tag", "keep_id": 1, "merge_id": 2}
        )
        merged = await client.post(
            "/api/vocabulary/merge", json={"kind": "tag", "keep_id": 3, "merge_id": 4}
        )
        same_entry = await client.post(
            "/api/vocabulary/merge", json={"kind": "tag", "keep_id": 3, "merge_id": 3}
        )
        after = await client.get("/api/vocabulary")
        reviewed_again = await client.post("/api/vocabulary/review")

    assert empty.json() == {"review": None, "dismissed_count": 0}
    assert [item["key"] for item in reviewed.json()["review"]["suggestions"]] == [
        "tag:3:4",
        "tag:1:2",
    ]
    assert reviewed.json()["review"]["reviewed_at"].startswith("2023-11-14")
    assert [item["key"] for item in dismissed.json()["review"]["suggestions"]] == ["tag:3:4"]
    assert merged.status_code == 200 and merges == [("tag", 3, 4)]
    assert same_entry.status_code == 422
    assert after.json()["review"]["suggestions"] == []
    assert after.json()["dismissed_count"] == 1
    # A dismissed pair stays hidden on later reviews; the merged pair returns only because the fake still reports it.
    assert [item["key"] for item in reviewed_again.json()["review"]["suggestions"]] == ["tag:3:4"]
