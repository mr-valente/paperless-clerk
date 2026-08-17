import asyncio
from pathlib import Path

import pytest

from paperless_clerk.clients.openai_compatible import ModelError
from paperless_clerk.config import Settings
from paperless_clerk.db import Database
from paperless_clerk.processing import DocumentProcessor


class FakeRenderer:
    def __init__(self, pages: int):
        self.page_count = pages
        self.rendered: list[int] = []

    def render(self, page_index: int) -> bytes:
        self.rendered.append(page_index)
        return f"page-{page_index + 1}".encode()

    async def render_async(self, page_index: int) -> bytes:
        return self.render(page_index)


class BoundedFakeModel:
    def __init__(self):
        self.active = 0
        self.maximum_active = 0
        self.called: list[int] = []

    async def ocr_page(self, image: bytes, *, page_number: int) -> str:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.called.append(page_number)
        await asyncio.sleep(0.001)
        self.active -= 1
        return f"Recognized complete text for page {page_number} with several useful words."


class OnePageFailureModel(BoundedFakeModel):
    async def ocr_page(self, image: bytes, *, page_number: int) -> str:
        if page_number == 3:
            raise ModelError("vision endpoint rejected this page", retryable=False)
        return await super().ocr_page(image, page_number=page_number)


@pytest.mark.asyncio
async def test_120_page_ocr_respects_concurrency_and_persists_each_page(tmp_path: Path) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(500, "ocr", 3)
    settings = Settings(page_concurrency=3)
    renderer = FakeRenderer(120)
    model = BoundedFakeModel()

    await DocumentProcessor(db, settings)._ocr_pages(job, renderer, model)  # noqa: SLF001

    assert model.maximum_active <= 3
    assert model.maximum_active > 1
    assert len(db.page_results(job["id"])) == 120
    assert all(row["status"] == "complete" for row in db.page_results(job["id"]))


@pytest.mark.asyncio
async def test_resumed_large_document_skips_completed_pages(tmp_path: Path) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(501, "ocr", 3)
    for page in range(1, 91):
        db.upsert_page(job["id"], page, "complete", 1, text=f"already finished {page}")
    renderer = FakeRenderer(100)
    model = BoundedFakeModel()

    await DocumentProcessor(db, Settings(page_concurrency=4))._ocr_pages(job, renderer, model)  # noqa: SLF001

    assert model.called == list(range(91, 101))
    assert renderer.rendered == list(range(90, 100))
    assert len(db.page_results(job["id"])) == 100


@pytest.mark.asyncio
async def test_individual_page_failure_does_not_cancel_siblings(tmp_path: Path) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(502, "ocr", 3)

    await DocumentProcessor(db, Settings(page_concurrency=3))._ocr_pages(  # noqa: SLF001
        job, FakeRenderer(6), OnePageFailureModel()
    )

    rows = db.page_results(job["id"])
    assert [row["status"] for row in rows] == [
        "complete",
        "complete",
        "failed",
        "complete",
        "complete",
        "complete",
    ]
