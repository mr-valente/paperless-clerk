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
    async def ocr_page(self, image: bytes, *, page_number: int) -> str:
        return (
            "Acme issued this detailed monthly statement with account totals and payment history."
        )


class NearMatchingOCRModel:
    async def ocr_page(self, image: bytes, *, page_number: int) -> str:
        return "Acme issued this detailed monthly statement with account totals and payment history"


class DivergentOCRModel:
    async def ocr_page(self, image: bytes, *, page_number: int) -> str:
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
    async def ocr_page(self, image: bytes, *, page_number: int) -> str:
        return FOOTER_BODY


class OCRCorrectedDuringRunPaperless:
    def __init__(self):
        self.document = {
            "id": 89,
            "title": "Corrected statement",
            "content": (
                "Acme issued this detailed monthly statement with account totals and payment history."
            ),
            "modified": "2026-08-13T12:00:00Z",
            "tags": [],
            "versions": [{"id": 89, "version_label": None, "is_root": True, "checksum": "source"}],
        }
        self.patches: list[tuple[int | None, dict]] = []
        self.version_labels: list[tuple[int, str]] = []
        self.uploads = 0
        self.downloads = 0

    async def download_document(self, document_id: int, destination: Path) -> str:
        self.downloads += 1
        return "same-source-hash"

    async def get_document(self, document_id: int) -> dict:
        return {**self.document, "versions": [dict(item) for item in self.document["versions"]]}

    async def upload_document_version(self, *args: object, **kwargs: object) -> str:
        self.uploads += 1
        return "version-task-89"

    async def wait_for_task(self, task_id: str, **_: object) -> dict:
        assert task_id == "version-task-89"
        if self.document["versions"][0]["id"] != 189:
            self.document["versions"].insert(
                0,
                {
                    "id": 189,
                    "version_label": "Paperless Clerk OCR",
                    "is_root": False,
                    "checksum": "source",
                },
            )
        return {"status": "success", "result_data": {"document_id": 189}}

    async def update_version_label(
        self, document_id: int, version_id: int, version_label: str
    ) -> dict:
        self.version_labels.append((version_id, version_label))
        for version in self.document["versions"]:
            if version["id"] == version_id:
                version["version_label"] = version_label
                return dict(version)
        raise AssertionError("version not found")

    async def update_document(
        self, document_id: int, patch: dict, *, version_id: int | None = None
    ) -> dict:
        self.patches.append((version_id, patch))
        self.document.update(patch)
        self.document["modified"] = "2026-08-13T12:02:00Z"
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
            "original_file_name": "statement.pdf",
            "archived_file_name": "statement-archive.pdf",
            "versions": [{"id": 92, "version_label": None, "is_root": True, "checksum": "source"}],
        }
        self.patches: list[dict] = []
        self.version_patches: list[tuple[int | None, dict]] = []
        self.version_labels: list[tuple[int, str]] = []
        self.uploads: list[dict] = []
        self.version_contents = {92: self.document["content"]}

    async def download_document(self, document_id: int, destination: Path) -> str:
        assert document_id == 92
        return "stable-source-hash"

    async def get_document(self, document_id: int) -> dict:
        assert document_id == 92
        return {**self.document, "versions": [dict(item) for item in self.document["versions"]]}

    async def update_document(
        self, document_id: int, patch: dict, *, version_id: int | None = None
    ) -> dict:
        assert document_id == 92
        self.patches.append(patch)
        self.version_patches.append((version_id, patch))
        if "content" in patch:
            target_version_id = version_id or self.document["versions"][0]["id"]
            self.version_contents[target_version_id] = patch["content"]
        self.document.update(patch)
        self.document["modified"] = "2026-08-13T12:01:00Z"
        return dict(self.document)

    async def upload_document_version(
        self,
        document_id: int,
        source: Path,
        *,
        filename: str,
        version_label: str,
    ) -> str:
        assert document_id == 92
        assert filename == "statement-archive.pdf"
        self.uploads.append({"source": source.name, "version_label": version_label})
        return "version-task-92"

    async def wait_for_task(self, task_id: str, **_: object) -> dict:
        assert task_id == "version-task-92"
        if self.document["versions"][0]["id"] != 192:
            self.document["versions"].insert(
                0,
                {
                    "id": 192,
                    "version_label": "Paperless Clerk OCR",
                    "is_root": False,
                    "checksum": "source",
                },
            )
            self.version_contents[192] = "Paperless re-extracted text"
            self.document["content"] = self.version_contents[192]
            self.document["modified"] = "2026-08-13T12:00:30Z"
        return {"status": "success", "result_data": {"document_id": 192}}

    async def update_version_label(
        self, document_id: int, version_id: int, version_label: str
    ) -> dict:
        assert document_id == 92
        self.version_labels.append((version_id, version_label))
        for version in self.document["versions"]:
            if version["id"] == version_id:
                version["version_label"] = version_label
                return dict(version)
        raise AssertionError("version not found")

    async def ensure_tag(self, name: str) -> dict:
        assert name == "ocr-conflict"
        return {"id": 99, "name": name}


class OCRFooterPaperless(OCRPreferencePaperless):
    def __init__(self):
        super().__init__()
        self.document["content"] = FOOTER_BODY + FOOTER_TEXT
        self.version_contents[92] = self.document["content"]


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
async def test_ocr_publishing_backs_up_text_added_while_clerk_was_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(89, "ocr", 3)
    paperless = OCRCorrectedDuringRunPaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, _, _ = await DocumentProcessor(db, Settings())._process_ocr(  # noqa: SLF001
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
    assert text.startswith("--- Page 1 ---\nAcme issued this detailed monthly statement")
    assert paperless.patches == [(189, {"content": text})]
    assert paperless.version_labels == [(89, "Pre-Clerk OCR backup")]
    assert paperless.uploads == 1
    assert paperless.downloads == 3


@pytest.mark.asyncio
async def test_existing_ocr_becomes_a_backup_version_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRPreferencePaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, pages, updated, _ = await DocumentProcessor(db, Settings())._process_ocr(  # noqa: SLF001
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
    assert pages == [
        (1, "Acme issued this detailed monthly statement with account totals and payment history")
    ]
    assert paperless.patches == [{"content": generated}]
    assert paperless.version_patches == [(192, {"content": generated})]
    assert paperless.version_contents[92].endswith("payment history.")
    assert paperless.version_contents[192] == generated
    assert paperless.version_labels == [(92, "Pre-Clerk OCR backup")]
    assert paperless.uploads == [{"source": "document", "version_label": "Paperless Clerk OCR"}]
    assert updated["content"] == generated
    events = db.get_job(job["id"], include_events=True)["events"]
    assert "ocr_version_queued" in {event["event_type"] for event in events}
    assert "ocr_version_published" in {event["event_type"] for event in events}
    checkpoint = db.get_job(job["id"])
    assert checkpoint["ocr_version_task_id"] == "version-task-92"
    assert checkpoint["ocr_version_id"] == 192


@pytest.mark.asyncio
async def test_existing_ocr_is_replaced_without_a_version_when_retention_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRPreferencePaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(
        db, Settings(keep_original_version=False)
    )._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        NearMatchingOCRModel(),  # type: ignore[arg-type]
    )

    assert outcome and outcome.status == "completed"
    assert updated["content"] == text
    assert paperless.version_patches == [(None, {"content": text})]
    assert paperless.version_contents[92] == text
    assert paperless.uploads == []
    assert paperless.version_labels == []
    checkpoint = db.get_job(job["id"])
    assert checkpoint and checkpoint["ocr_version_task_id"] is None
    events = db.get_job(job["id"], include_events=True)["events"]
    published = next(event for event in events if event["event_type"] == "ocr_published")
    assert published["data"]["replaced_existing"] is True


@pytest.mark.asyncio
async def test_existing_clerk_version_is_reused_without_another_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRPreferencePaperless()
    paperless.document["versions"].insert(
        0,
        {
            "id": 192,
            "version_label": "Paperless Clerk OCR",
            "is_root": False,
            "checksum": "source",
        },
    )
    paperless.version_contents[192] = paperless.document["content"]
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, _, _ = await DocumentProcessor(db, Settings())._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        NearMatchingOCRModel(),  # type: ignore[arg-type]
    )

    assert outcome and outcome.status == "completed"
    assert paperless.uploads == []
    assert paperless.version_labels == []
    assert paperless.version_patches == [(192, {"content": text})]


@pytest.mark.asyncio
async def test_pending_version_task_is_resumed_without_another_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    queued, _ = db.enqueue_job(92, "ocr", 3)
    db.set_ocr_version_task(queued["id"], "version-task-92")
    job = db.get_job(queued["id"])
    assert job is not None
    paperless = OCRPreferencePaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    await DocumentProcessor(db, Settings(keep_original_version=False))._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        NearMatchingOCRModel(),  # type: ignore[arg-type]
    )

    assert paperless.uploads == []
    assert db.get_job(job["id"])["ocr_version_id"] == 192


@pytest.mark.asyncio
async def test_interrupted_version_upload_is_not_repeated_while_outcome_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    queued, _ = db.enqueue_job(92, "ocr", 3)
    db.set_ocr_version_task(queued["id"], processing.VERSION_UPLOAD_STARTED)
    job = db.get_job(queued["id"])
    assert job is not None
    paperless = OCRPreferencePaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    with pytest.raises(ProcessingError, match="Inspect the document's version history") as raised:
        await DocumentProcessor(db, Settings())._process_ocr(  # noqa: SLF001
            job,
            dict(paperless.document),
            paperless,  # type: ignore[arg-type]
            NearMatchingOCRModel(),  # type: ignore[arg-type]
        )

    assert raised.value.code == "ocr_version_upload_ambiguous"
    assert paperless.uploads == []
    assert db.get_job(job["id"])["ocr_version_task_id"] == processing.VERSION_UPLOAD_STARTED


@pytest.mark.asyncio
async def test_failed_paperless_version_task_clears_checkpoint_for_manual_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedVersionPaperless(OCRPreferencePaperless):
        async def wait_for_task(self, task_id: str, **_: object) -> dict:
            assert task_id == "version-task-92"
            return {
                "status": "failure",
                "result_data": {"error_message": "consumer rejected the file"},
            }

    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = FailedVersionPaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    with pytest.raises(ProcessingError, match="consumer rejected") as raised:
        await DocumentProcessor(db, Settings())._process_ocr(  # noqa: SLF001
            job,
            dict(paperless.document),
            paperless,  # type: ignore[arg-type]
            NearMatchingOCRModel(),  # type: ignore[arg-type]
        )

    assert raised.value.code == "ocr_version_failed"
    checkpoint = db.get_job(job["id"])
    assert checkpoint and checkpoint["ocr_version_task_id"] is None
    assert checkpoint["ocr_version_id"] is None


@pytest.mark.asyncio
async def test_deepseek_can_publish_versioned_clerk_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vllm_profiles: None
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRPreferencePaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(
        db, Settings(ocr_profile="deepseek_ocr")
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
    assert "ocr_version_published" in {event["event_type"] for event in events}


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["deepseek_ocr", "glm_ocr"])
async def test_specialist_ocr_without_baseline_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vllm_profiles: None, profile: str
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = EmptyOCRPaperless()
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(
        db, Settings(ocr_profile=profile)
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
    assert configuration["data"]["profile"] == profile
    assert configuration["data"]["prompt"] == (
        "Free OCR." if profile == "deepseek_ocr" else "Text Recognition:"
    )
    assert ("vllm_xargs" in configuration["data"]["extra_body"]) is (profile == "deepseek_ocr")


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
async def test_shorter_clerk_ocr_becomes_default_while_footer_version_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vllm_profiles: None
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRFooterPaperless()
    original = paperless.document["content"]
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(
        db, Settings(ocr_profile="deepseek_ocr")
    )._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        FooterOmittingOCRModel(),  # type: ignore[arg-type]
    )

    assert outcome and outcome.status == "completed"
    assert text != original
    assert updated["content"] == text
    assert paperless.patches == [{"content": text}]
    assert paperless.version_contents[92] == original
    assert paperless.version_contents[192] == text
    events = db.get_job(job["id"], include_events=True)["events"]
    configuration = next(event for event in events if event["event_type"] == "ocr_configuration")
    assert configuration["data"]["model"] == "qwen2.5vl:7b"
    assert configuration["data"]["profile"] == "deepseek_ocr"
    assert configuration["data"]["prompt"] == "Free OCR."
    assert configuration["data"]["max_output_tokens"] == 4096


@pytest.mark.asyncio
async def test_divergent_existing_ocr_is_backed_up_without_intervention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    job, _ = db.enqueue_job(92, "ocr", 3)
    paperless = OCRPreferencePaperless()
    original = paperless.document["content"]
    monkeypatch.setattr(processing, "DocumentRenderer", SinglePageRenderer)

    outcome, text, _, updated, _ = await DocumentProcessor(db, Settings())._process_ocr(  # noqa: SLF001
        job,
        dict(paperless.document),
        paperless,  # type: ignore[arg-type]
        DivergentOCRModel(),  # type: ignore[arg-type]
    )

    assert outcome and outcome.status == "completed"
    assert outcome.phase == "complete"
    assert text.startswith("--- Page 1 ---\nNorthwind provided")
    assert updated["content"] == text
    assert paperless.version_contents[92] == original
    assert db.list_conflicts() == []
