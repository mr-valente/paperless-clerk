from pathlib import Path

import pytest

import paperless_clerk.processing as processing
from paperless_clerk.clients.paperless import PaperlessError
from paperless_clerk.config import Settings
from paperless_clerk.db import Database
from paperless_clerk.processing import ProcessingError


class FakePaperless:
    def __init__(self, _: Settings):
        self.patches: list[dict] = []
        self.document = {
            "id": 71,
            "title": "Disputed statement",
            "tags": [5, 9],
            "content": "existing complete text",
            "modified": "2026-08-13T10:00:00Z",
        }

    async def get_document(self, _: int) -> dict:
        return self.document

    async def update_document(self, _: int, patch: dict) -> dict:
        self.patches.append(patch)
        self.document = {**self.document, **patch, "modified": "2026-08-13T11:00:00Z"}
        return self.document

    async def close(self) -> None:
        return None


class FailingPaperless(FakePaperless):
    async def get_document(self, _: int) -> dict:
        raise PaperlessError("Paperless is temporarily unavailable", retryable=True)


class EmptyPaperless(FakePaperless):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.document["content"] = ""


class EditedPaperless(FakePaperless):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.document["content"] = (
            "Human-corrected Paperless OCR with enough meaningful text to preserve."
        )


@pytest.mark.asyncio
async def test_conflict_resolution_closes_review_job_and_queues_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(71, "full", 3)
    db.mark_needs_review(job["id"], "ocr_conflict", "review")
    conflict = db.create_conflict(
        job_id=job["id"],
        document_id=71,
        document_title="Disputed statement",
        existing_text="existing complete text",
        generated_text="generated complete text",
        score=0.31,
        metrics={"score": 0.31},
        diff=[{"operation": "replace", "existing": "existing", "generated": "generated"}],
        tag_id=9,
    )
    assert db.retry_job(job["id"]) is None
    monkeypatch.setattr(processing, "PaperlessClient", FakePaperless)

    result = await processing.resolve_conflict(
        database=db,
        settings=Settings(),
        conflict_id=conflict["id"],
        resolution="use_clerk",
    )

    resolved = db.get_conflict(conflict["id"])
    assert resolved["status"] == "resolved"
    assert resolved["existing_text"] == resolved["generated_text"] == ""
    assert db.get_job(job["id"])["status"] == "completed"
    assert result["job"]["mode"] == "metadata"
    assert result["job"]["status"] == "queued"


def test_conflict_resolution_claim_is_atomic_and_recoverable(tmp_path: Path) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(72, "full", 3)
    conflict = db.create_conflict(
        job_id=job["id"],
        document_id=72,
        document_title="Double click",
        existing_text="existing",
        generated_text="generated",
        score=0.3,
        metrics={"score": 0.3},
        diff=[],
        tag_id=9,
    )

    assert db.claim_conflict_resolution(conflict["id"], "keep_existing") is True
    assert db.claim_conflict_resolution(conflict["id"], "use_clerk") is False

    db.release_conflict_resolution(conflict["id"], "keep_existing")

    assert db.claim_conflict_resolution(conflict["id"], "use_clerk") is True


def test_historic_job_cannot_be_retried_around_an_open_conflict(tmp_path: Path) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    historic, _ = db.enqueue_job(73, "full", 1)
    claimed = db.claim_job("worker", 60)
    assert claimed
    db.fail_or_retry(historic["id"], "failed", "failed", False)
    review, _ = db.enqueue_job(73, "full", 1)
    db.mark_needs_review(review["id"], "ocr_conflict", "review")
    db.create_conflict(
        job_id=review["id"],
        document_id=73,
        document_title="Needs review",
        existing_text="existing",
        generated_text="generated",
        score=0.3,
        metrics={"score": 0.3},
        diff=[],
        tag_id=9,
    )

    assert db.retry_job(historic["id"]) is None


@pytest.mark.asyncio
async def test_failed_paperless_resolution_releases_atomic_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(74, "full", 3)
    db.mark_needs_review(job["id"], "ocr_conflict", "review")
    conflict = db.create_conflict(
        job_id=job["id"],
        document_id=74,
        document_title="Retry resolution",
        existing_text="existing",
        generated_text="generated",
        score=0.3,
        metrics={"score": 0.3},
        diff=[],
        tag_id=9,
    )
    monkeypatch.setattr(processing, "PaperlessClient", FailingPaperless)

    with pytest.raises(ProcessingError) as raised:
        await processing.resolve_conflict(
            database=db,
            settings=Settings(),
            conflict_id=conflict["id"],
            resolution="keep_existing",
        )

    assert raised.value.code == "paperless_error"
    assert db.claim_conflict_resolution(conflict["id"], "keep_existing") is True


@pytest.mark.asyncio
async def test_empty_paperless_baseline_cannot_be_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(75, "full", 3)
    db.mark_needs_review(job["id"], "ocr_conflict", "review")
    conflict = db.create_conflict(
        job_id=job["id"],
        document_id=75,
        document_title="No OCR baseline",
        existing_text="",
        generated_text="generated complete text with enough meaningful content for review",
        score=0.0,
        metrics={"score": 0.0},
        diff=[],
        tag_id=9,
    )
    paperless = EmptyPaperless(Settings())
    monkeypatch.setattr(processing, "PaperlessClient", lambda _: paperless)

    with pytest.raises(ProcessingError) as raised:
        await processing.resolve_conflict(
            database=db,
            settings=Settings(),
            conflict_id=conflict["id"],
            resolution="keep_existing",
        )

    assert raised.value.code == "no_existing_ocr"
    assert paperless.patches == []
    assert db.get_conflict(conflict["id"])["status"] == "open"
    assert db.get_job(job["id"])["status"] == "needs_review"
    assert db.claim_conflict_resolution(conflict["id"], "keep_existing") is True
    db.release_conflict_resolution(conflict["id"], "keep_existing")


@pytest.mark.asyncio
async def test_clerk_resolution_does_not_overwrite_newer_paperless_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(76, "full", 3)
    db.mark_needs_review(job["id"], "ocr_conflict", "review")
    conflict = db.create_conflict(
        job_id=job["id"],
        document_id=76,
        document_title="Human-edited OCR",
        existing_text="existing complete text",
        generated_text="generated complete text",
        score=0.3,
        metrics={"score": 0.3},
        diff=[],
        tag_id=9,
    )
    paperless = EditedPaperless(Settings())
    monkeypatch.setattr(processing, "PaperlessClient", lambda _: paperless)

    with pytest.raises(ProcessingError) as raised:
        await processing.resolve_conflict(
            database=db,
            settings=Settings(),
            conflict_id=conflict["id"],
            resolution="use_clerk",
        )

    assert raised.value.code == "conflict_source_changed"
    assert paperless.patches == []
    assert db.get_conflict(conflict["id"])["status"] == "open"
    assert db.get_job(job["id"])["status"] == "needs_review"


@pytest.mark.asyncio
async def test_added_paperless_ocr_can_resolve_an_empty_baseline_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(77, "full", 3)
    db.mark_needs_review(job["id"], "ocr_conflict", "review")
    conflict = db.create_conflict(
        job_id=job["id"],
        document_id=77,
        document_title="Paperless OCR added later",
        existing_text="",
        generated_text="generated complete text",
        score=0.0,
        metrics={"score": 0.0},
        diff=[],
        tag_id=9,
    )
    paperless = EditedPaperless(Settings())
    monkeypatch.setattr(processing, "PaperlessClient", lambda _: paperless)

    result = await processing.resolve_conflict(
        database=db,
        settings=Settings(),
        conflict_id=conflict["id"],
        resolution="keep_existing",
    )

    assert result["resolution"] == "keep_existing"
    assert paperless.patches == [{"tags": [5]}]
    assert paperless.document["content"].startswith("Human-corrected")
    assert db.get_conflict(conflict["id"])["status"] == "resolved"
    assert result["job"]["mode"] == "metadata"
