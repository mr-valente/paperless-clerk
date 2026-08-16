from __future__ import annotations

from pathlib import Path

import pytest

import paperless_clerk.processing as processing
from paperless_clerk.config import Settings
from paperless_clerk.db import Database
from paperless_clerk.processing import DocumentProcessor, ProcessingError, _without_watch_tag
from paperless_clerk.schemas import MetadataProposal


class NoopMetadataAnalyzer:
    def __init__(self, settings: Settings, model: object):
        pass

    async def analyze(self, **_: object) -> tuple[MetadataProposal, dict]:
        return MetadataProposal(), {
            "chunks": [{"index": 0, "page_start": 1, "page_end": 1}],
            "candidate_counts": {
                "tags": 0,
                "correspondents": 0,
                "document_types": 0,
                "custom_fields": 0,
            },
            "candidate_ids": {
                "tag": [],
                "correspondent": [],
                "document_type": [],
                "custom_field": [],
            },
            "model_diagnostics": [
                {"stage": "focused_tag_review", "status": "completed", "returned_tag_count": 0}
            ],
        }


class ChangedDocumentPaperless:
    async def catalog(self) -> dict[str, list]:
        return {"tags": [], "correspondents": [], "document_types": [], "custom_fields": []}

    async def get_document(self, document_id: int) -> dict:
        return {
            "id": document_id,
            "title": "Edited while Clerk was working",
            "content": "New Paperless OCR text",
            "tags": [],
            "custom_fields": [],
        }


class SinglePageRenderer:
    page_count = 1

    def __init__(self, *_: object, **__: object):
        pass

    def render(self, page_index: int) -> bytes:
        return b"page image"

    async def render_async(self, page_index: int) -> bytes:
        return self.render(page_index)

    def __enter__(self) -> SinglePageRenderer:
        return self

    def __exit__(self, *_: object) -> None:
        pass


class MatchingOCRModel:
    async def ocr_page(self, image: bytes, *, page_number: int, prompt: str) -> str:
        return (
            "Acme issued this detailed monthly statement with account totals and payment history."
        )


class NearMatchingOCRModel:
    async def ocr_page(self, image: bytes, *, page_number: int, prompt: str) -> str:
        return "Acme issued this detailed monthly statement with account totals and payment history"


class DivergentOCRModel:
    async def ocr_page(self, image: bytes, *, page_number: int, prompt: str) -> str:
        return "Northwind provided an unrelated shipping notice for a different household delivery."


FOOTER_BODY = (
    "Charles Schwab updated your contact information and asks you to review the "
    "account details online. Thank you for investing with Schwab. "
) * 8
FOOTER_TEXT = (
    "Brokerage products are not FDIC insured and may lose value. "
    "Deposit products are offered by Charles Schwab Bank Member FDIC."
)


class FooterOmittingOCRModel:
    async def ocr_page(self, image: bytes, *, page_number: int, prompt: str) -> str:
        return FOOTER_BODY


class OCRCorrectedDuringRunPaperless:
    def __init__(self):
        self.patches: list[dict] = []
        self.downloads = 0

    async def download_document(self, document_id: int, destination: Path) -> str:
        self.downloads += 1
        return "same-source-hash"

    async def get_document(self, document_id: int) -> dict:
        return {
            "id": document_id,
            "title": "Corrected statement",
            "content": (
                "Acme issued this detailed monthly statement with account totals and payment history."
            ),
            "modified": "2026-08-13T12:00:00Z",
            "tags": [],
        }

    async def update_document(self, document_id: int, patch: dict) -> dict:
        self.patches.append(patch)
        return await self.get_document(document_id)


class OCRPreferencePaperless:
    def __init__(self):
        self.document = {
            "id": 92,
            "title": "Existing statement",
            "content": (
                "Acme issued this detailed monthly statement with account totals and payment history."
            ),
            "modified": "2026-08-13T12:00:00Z",
            "tags": [],
        }
        self.patches: list[dict] = []

    async def download_document(self, document_id: int, destination: Path) -> str:
        assert document_id == 92
        return "stable-source-hash"

    async def get_document(self, document_id: int) -> dict:
        assert document_id == 92
        return dict(self.document)

    async def update_document(self, document_id: int, patch: dict) -> dict:
        assert document_id == 92
        self.patches.append(patch)
        self.document.update(patch)
        self.document["modified"] = "2026-08-13T12:01:00Z"
        return dict(self.document)

    async def ensure_tag(self, name: str) -> dict:
        assert name == "ocr-conflict"
        return {"id": 99, "name": name}


class OCRFooterPaperless(OCRPreferencePaperless):
    def __init__(self):
        super().__init__()
        self.document["content"] = FOOTER_BODY + FOOTER_TEXT


class EmptyOCRPaperless(OCRPreferencePaperless):
    def __init__(self):
        super().__init__()
        self.document["content"] = ""


class WatchTagPaperless:
    def __init__(self):
        self.document = {
            "id": 91,
            "title": "Employment agreement",
            "content": "A sufficiently detailed employment agreement for metadata analysis.",
            "modified": "2026-08-13T12:00:00Z",
            "tags": [7, 55],
            "custom_fields": [],
        }
        self.patches: list[dict] = []

    async def catalog(self) -> dict[str, list]:
        return {
            "tags": [
                {"id": 7, "name": "Employment"},
                {"id": 55, "name": "Clerk Inbox"},
            ],
            "correspondents": [],
            "document_types": [],
            "custom_fields": [],
        }

    async def get_document(self, document_id: int) -> dict:
        assert document_id == 91
        return dict(self.document)

    async def update_document(self, document_id: int, patch: dict) -> dict:
        assert document_id == 91
        self.patches.append(patch)
        self.document.update(patch)
        self.document["modified"] = "2026-08-13T12:01:00Z"
        return dict(self.document)


def test_watch_tag_is_hidden_from_metadata_document_and_vocabulary() -> None:
    document = {"id": 90, "tags": [7, 55]}
    catalog = {
        "tags": [{"id": 7, "name": "Employment"}, {"id": 55, "name": "Clerk Inbox"}],
        "correspondents": [],
        "document_types": [],
        "custom_fields": [],
    }

    model_document, model_catalog, watch_tag = _without_watch_tag(
        document, catalog, "  clerk   inbox "
    )

    assert model_document["tags"] == [7]
    assert [tag["id"] for tag in model_catalog["tags"]] == [7]
    assert watch_tag == {"id": 55, "name": "Clerk Inbox"}
    assert document["tags"] == [7, 55]
    assert len(catalog["tags"]) == 2


@pytest.mark.asyncio
async def test_successful_metadata_consumes_watch_tag_after_hiding_it_from_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class CapturingAnalyzer(NoopMetadataAnalyzer):
        async def analyze(self, **values: object) -> tuple[MetadataProposal, dict]:
            captured.update(values)
            return await super().analyze(**values)

    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(91, "metadata", 3)
    paperless = WatchTagPaperless()
    monkeypatch.setattr(processing, "MetadataAnalyzer", CapturingAnalyzer)

    updated = await DocumentProcessor(db, Settings(automation_tag="Clerk Inbox"))._process_metadata(  # noqa: SLF001
        job,
        dict(paperless.document),
        [(1, paperless.document["content"])],
        paperless,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
    )

    assert captured["document"]["tags"] == [7]
    assert [tag["id"] for tag in captured["catalog"]["tags"]] == [7]
    assert paperless.patches == [{"tags": [7]}]
    assert updated["tags"] == [7]
    decision = db.get_decision(db.list_decisions()[0]["id"])
    assert decision is not None
    assert decision["applied"]["removed"][0]["id"] == 55


@pytest.mark.asyncio
async def test_tagless_metadata_decision_is_explicit_and_observable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(91, "metadata", 3)
    paperless = WatchTagPaperless()
    paperless.document["tags"] = [55]
    monkeypatch.setattr(processing, "MetadataAnalyzer", NoopMetadataAnalyzer)

    await DocumentProcessor(db, Settings(automation_tag="Clerk Inbox"))._process_metadata(  # noqa: SLF001
        job,
        dict(paperless.document),
        [(1, paperless.document["content"])],
        paperless,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
    )

    decision = db.get_decision(db.list_decisions()[0]["id"])
    assert decision is not None
    assert decision["status"] == "no_tags"
    assert decision["rationale"]["tag_review"]["outcome"] == "no_tags"
    assert decision["rationale"]["candidate_ids"] == {
        "tag": [],
        "correspondent": [],
        "document_type": [],
        "custom_field": [],
    }
    assert decision["rationale"]["model_diagnostics"][0]["stage"] == "focused_tag_review"
    events = db.get_job(job["id"], include_events=True)["events"]
    assert "metadata_no_tags" in {event["event_type"] for event in events}


@pytest.mark.asyncio
async def test_metadata_write_retries_if_paperless_ocr_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(88, "metadata", 3)
    monkeypatch.setattr(processing, "MetadataAnalyzer", NoopMetadataAnalyzer)

    with pytest.raises(ProcessingError) as raised:
        await DocumentProcessor(db, Settings())._process_metadata(  # noqa: SLF001
            job,
            {
                "id": 88,
                "title": "Original",
                "content": "Original Paperless OCR text",
                "tags": [],
                "custom_fields": [],
            },
            [(1, "Original Paperless OCR text")],
            ChangedDocumentPaperless(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )

    assert raised.value.code == "document_changed"
    assert raised.value.retryable is True


@pytest.mark.asyncio
async def test_ocr_verification_uses_latest_paperless_text_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(89, "ocr", 3)
    paperless = OCRCorrectedDuringRunPaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, _, _ = await DocumentProcessor(
        db, Settings(prefer_clerk_ocr=False)
    )._process_ocr(  # noqa: SLF001
        job,
        {
            "id": 89,
            "title": "Initially empty",
            "content": "",
            "modified": "2026-08-13T11:00:00Z",
            "tags": [],
        },
        paperless,  # type: ignore[arg-type]
        MatchingOCRModel(),  # type: ignore[arg-type]
    )

    assert outcome and outcome.status == "completed"
    assert (
        text
        == "Acme issued this detailed monthly statement with account totals and payment history."
    )
    assert paperless.patches == []
    assert paperless.downloads == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prefer_clerk", "expected_source"),
    [(True, "clerk"), (False, "paperless")],
)
async def test_trusted_ocr_match_uses_configured_source_preference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefer_clerk: bool,
    expected_source: str,
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRPreferencePaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, pages, updated, _ = await DocumentProcessor(
        db, Settings(prefer_clerk_ocr=prefer_clerk)
    )._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        NearMatchingOCRModel(),  # type: ignore[arg-type]
    )

    generated = (
        "--- Page 1 ---\n"
        "Acme issued this detailed monthly statement with account totals and payment history\n"
    )
    existing = (
        "Acme issued this detailed monthly statement with account totals and payment history."
    )
    assert outcome and outcome.status == "completed"
    assert text == (generated if prefer_clerk else existing)
    assert pages == (
        [(1, "Acme issued this detailed monthly statement with account totals and payment history")]
        if prefer_clerk
        else [(1, existing)]
    )
    assert paperless.patches == ([{"content": generated}] if prefer_clerk else [])
    assert updated["content"] == (generated if prefer_clerk else existing)
    events = db.get_job(job["id"], include_events=True)["events"]
    comparison = next(event for event in events if event["event_type"] == "ocr_compared")
    assert comparison["data"]["selected_source"] == expected_source
    assert ("ocr_preference_applied" in {event["event_type"] for event in events}) is prefer_clerk


@pytest.mark.asyncio
async def test_deepseek_match_can_publish_preferred_clerk_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRPreferencePaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(
        db, Settings(prefer_clerk_ocr=True, ocr_profile="deepseek_ocr")
    )._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        NearMatchingOCRModel(),  # type: ignore[arg-type]
    )

    generated = (
        "--- Page 1 ---\n"
        "Acme issued this detailed monthly statement with account totals and payment history\n"
    )
    assert outcome and outcome.status == "completed"
    assert text == generated
    assert updated["content"] == generated
    assert paperless.patches == [{"content": generated}]
    events = db.get_job(job["id"], include_events=True)["events"]
    comparison = next(event for event in events if event["event_type"] == "ocr_compared")
    assert comparison["data"]["selected_source"] == "clerk"
    assert "ocr_preference_applied" in {event["event_type"] for event in events}


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["deepseek_ocr", "glm_ocr"])
async def test_specialist_ocr_without_baseline_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = EmptyOCRPaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(
        db, Settings(prefer_clerk_ocr=True, ocr_profile=profile)
    )._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        MatchingOCRModel(),  # type: ignore[arg-type]
    )

    assert outcome and outcome.status == "completed"
    assert text == updated["content"]
    assert "Acme issued this detailed monthly statement" in text
    assert paperless.patches == [{"content": text}]
    assert db.list_conflicts() == []
    events = db.get_job(job["id"], include_events=True)["events"]
    configuration = next(event for event in events if event["event_type"] == "ocr_configuration")
    assert configuration["data"]["image_format"] == "jpeg"
    assert configuration["data"]["prompt"] == (
        "Free OCR." if profile == "deepseek_ocr" else "Text Recognition:"
    )
    assert configuration["data"]["decoding"] == (
        {"temperature": 0, "top_k": 1} if profile == "deepseek_ocr" else {"temperature": 0}
    )


@pytest.mark.asyncio
async def test_ordinary_ocr_without_baseline_still_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = EmptyOCRPaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(
        db, Settings(ocr_profile="generic")
    )._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        MatchingOCRModel(),  # type: ignore[arg-type]
    )

    assert outcome and outcome.status == "completed"
    assert text == updated["content"]
    assert "Acme issued this detailed monthly statement" in text
    assert paperless.patches == [{"content": text}]
    assert db.list_conflicts() == []


@pytest.mark.asyncio
async def test_less_complete_clerk_ocr_never_overwrites_an_existing_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRFooterPaperless()
    original = paperless.document["content"]
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(
        db, Settings(prefer_clerk_ocr=True, ocr_profile="deepseek_ocr")
    )._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        FooterOmittingOCRModel(),  # type: ignore[arg-type]
    )

    assert outcome and outcome.status == "completed"
    assert text == original
    assert updated["content"] == original
    assert paperless.patches == []
    events = db.get_job(job["id"], include_events=True)["events"]
    comparison = next(event for event in events if event["event_type"] == "ocr_compared")
    assert comparison["data"]["selected_source"] == "paperless"
    assert comparison["data"]["coverage_safeguard"] is True
    assert any(event["event_type"] == "ocr_coverage_safeguard" for event in events)
    configuration = next(event for event in events if event["event_type"] == "ocr_configuration")
    assert configuration["data"]["model"] == "qwen2.5vl:7b"
    assert configuration["data"]["profile"] == "deepseek_ocr"
    assert configuration["data"]["image_format"] == "jpeg"
    assert configuration["data"]["prompt"] == "Free OCR."
    assert configuration["data"]["decoding"] == {"temperature": 0, "top_k": 1}
    assert "vllm_xargs" not in configuration["data"]
    assert "publication_policy" not in configuration["data"]


@pytest.mark.asyncio
async def test_clerk_preference_never_bypasses_low_match_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRPreferencePaperless()
    original = paperless.document["content"]
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(
        db, Settings(prefer_clerk_ocr=True)
    )._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        DivergentOCRModel(),  # type: ignore[arg-type]
    )

    assert outcome and outcome.status == "needs_review"
    assert outcome.phase == "ocr_conflict"
    assert text == original
    assert paperless.patches == [{"tags": [99]}]
    assert updated["content"] == original
    comparison = next(
        event
        for event in db.get_job(job["id"], include_events=True)["events"]
        if event["event_type"] == "ocr_compared"
    )
    assert comparison["data"]["selected_source"] == "manual_review"
