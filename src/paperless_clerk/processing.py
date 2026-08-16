from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperless_clerk.clients.ntfy import NtfyClient
from paperless_clerk.clients.openai_compatible import (
    DEEPSEEK_VLLM_XARGS,
    OCR_REQUEST_CONTRACT_VERSION,
    ModelError,
    OpenAICompatibleClient,
)
from paperless_clerk.clients.paperless import PaperlessClient, PaperlessError
from paperless_clerk.config import Settings, SettingsManager
from paperless_clerk.db import Database
from paperless_clerk.domain.chunking import pages_from_assembled_text
from paperless_clerk.domain.ocr_compare import assemble_pages, compare_ocr, meaningful_ocr
from paperless_clerk.metadata import MetadataAnalyzer, MetadataPlanner, apply_metadata_plan
from paperless_clerk.prompts import (
    DEEPSEEK_FREE_OCR_PAGE_PROMPT,
    DEEPSEEK_OCR_PAGE_PROMPT,
    OCR_PAGE_PROMPT,
)
from paperless_clerk.rendering import DocumentRenderer, RenderError

log = logging.getLogger(__name__)


def _normalized_tag_name(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _watch_tag_from_catalog(
    catalog: dict[str, list[dict[str, Any]]], name: str
) -> dict[str, Any] | None:
    target = _normalized_tag_name(name)
    if not target:
        return None
    return next(
        (tag for tag in catalog.get("tags", []) if _normalized_tag_name(tag.get("name")) == target),
        None,
    )


def _without_watch_tag(
    document: dict[str, Any], catalog: dict[str, list[dict[str, Any]]], name: str
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any] | None]:
    """Hide the workflow-only watch tag from model-controlled metadata inputs."""

    watch_tag = _watch_tag_from_catalog(catalog, name)
    if not watch_tag:
        return document, catalog, None
    watch_tag_id = int(watch_tag["id"])
    model_document = {
        **document,
        "tags": [int(tag_id) for tag_id in document.get("tags", []) if int(tag_id) != watch_tag_id],
    }
    model_catalog = {
        **catalog,
        "tags": [tag for tag in catalog.get("tags", []) if int(tag["id"]) != watch_tag_id],
    }
    return model_document, model_catalog, watch_tag


class ProcessingError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ProcessOutcome:
    status: str
    phase: str
    message: str = ""


class DocumentProcessor:
    def __init__(
        self,
        database: Database,
        settings: Settings,
        vocabulary_lock: asyncio.Lock | None = None,
    ):
        self.database = database
        self.settings = settings
        self.vocabulary_lock = vocabulary_lock or asyncio.Lock()

    async def process(self, job: dict[str, Any]) -> ProcessOutcome:
        paperless = PaperlessClient(self.settings)
        ocr_model = OpenAICompatibleClient(self.settings, "ocr")
        metadata_model = OpenAICompatibleClient(self.settings, "metadata")
        document: dict[str, Any] | None = None
        source_hash: str | None = job.get("source_hash")
        try:
            self.database.update_job(job["id"], phase="fetching_document")
            document = await paperless.get_document(int(job["document_id"]))
            title = str(document.get("title") or f"Document {job['document_id']}")
            self.database.update_job(job["id"], title=title)
            self.database.set_document_state(
                int(document["id"]), document.get("modified"), source_hash, job["id"], "running"
            )
            if self.database.has_open_conflict(int(document["id"])):
                return ProcessOutcome(
                    "needs_review",
                    "ocr_conflict",
                    "Resolve the existing OCR conflict before starting another processing run",
                )

            text_for_metadata = str(document.get("content") or "")
            pages_for_metadata: list[tuple[int, str]] | None = None
            updated_document = document

            if job["mode"] in {"full", "ocr"}:
                (
                    outcome,
                    text_for_metadata,
                    pages_for_metadata,
                    updated_document,
                    source_hash,
                ) = await self._process_ocr(job, document, paperless, ocr_model)
                if outcome is not None:
                    self.database.set_document_state(
                        int(document["id"]),
                        updated_document.get("modified"),
                        source_hash,
                        job["id"],
                        outcome.status,
                    )
                    return outcome

            if job["mode"] in {"full", "metadata"}:
                if not meaningful_ocr(text_for_metadata, self.settings.ocr_min_chars):
                    raise ProcessingError(
                        "no_text_for_metadata",
                        "Document has no meaningful OCR text for metadata analysis",
                    )
                pages_for_metadata = pages_for_metadata or pages_from_assembled_text(
                    text_for_metadata
                )
                updated_document = await self._process_metadata(
                    job, updated_document, pages_for_metadata, paperless, metadata_model
                )

            self.database.set_document_state(
                int(document["id"]),
                updated_document.get("modified"),
                source_hash,
                job["id"],
                "completed",
            )
            return ProcessOutcome("completed", "complete")
        except PaperlessError as exc:
            raise ProcessingError("paperless_error", str(exc), retryable=exc.retryable) from exc
        except ModelError as exc:
            raise ProcessingError("model_error", str(exc), retryable=exc.retryable) from exc
        except RenderError as exc:
            raise ProcessingError("render_error", str(exc)) from exc
        finally:
            await asyncio.gather(
                paperless.close(), ocr_model.close(), metadata_model.close(), return_exceptions=True
            )

    async def _process_ocr(
        self,
        job: dict[str, Any],
        document: dict[str, Any],
        paperless: PaperlessClient,
        model: OpenAICompatibleClient,
    ) -> tuple[
        ProcessOutcome | None,
        str,
        list[tuple[int, str]],
        dict[str, Any],
        str,
    ]:
        self.database.update_job(job["id"], phase="downloading")
        deepseek_vllm_profile = self.settings.ocr_profile == "deepseek_ocr"
        deepseek_llamacpp_profile = self.settings.ocr_profile == "deepseek_ocr_llamacpp"
        profile_prompt = (
            DEEPSEEK_OCR_PAGE_PROMPT
            if deepseek_vllm_profile
            else DEEPSEEK_FREE_OCR_PAGE_PROMPT
            if deepseek_llamacpp_profile
            else OCR_PAGE_PROMPT
        )
        image_format = "png" if deepseek_vllm_profile else "jpeg"
        request_configuration: dict[str, Any] = {
            "contract_version": OCR_REQUEST_CONTRACT_VERSION,
            "model": self.settings.ocr_model,
            "profile": self.settings.ocr_profile,
            "context_tokens": self.settings.ocr_context_tokens,
            "max_output_tokens": self.settings.ocr_max_output_tokens,
            "render_dpi": self.settings.render_dpi,
            "max_image_pixels": self.settings.max_image_pixels,
            "image_format": image_format,
            "prompt": profile_prompt,
            "decoding": {"temperature": 0},
        }
        if image_format == "jpeg":
            request_configuration["jpeg_quality"] = self.settings.jpeg_quality
        if deepseek_vllm_profile:
            request_configuration["skip_special_tokens"] = False
            request_configuration["vllm_xargs"] = dict(DEEPSEEK_VLLM_XARGS)
            request_configuration["publication_policy"] = "existing_paperless_or_review"
            request_configuration["completion_policy"] = (
                "discard_token_limited_or_repetition_stopped_output"
            )
        elif deepseek_llamacpp_profile:
            request_configuration["decoding"]["top_k"] = 1
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "base_url": self.settings.openai_base_url,
                    **request_configuration,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        previous_fingerprint = job.get("ocr_fingerprint")
        if previous_fingerprint and previous_fingerprint != fingerprint:
            self.database.clear_pages(job["id"])
            self.database.add_event(
                job["id"],
                "warning",
                "ocr_configuration_changed",
                "OCR model or rendering configuration changed; saved page results were discarded",
                request_configuration,
            )
        if previous_fingerprint != fingerprint:
            self.database.add_event(
                job["id"],
                "info",
                "ocr_configuration",
                f"Using OCR model {self.settings.ocr_model} with profile {self.settings.ocr_profile}",
                request_configuration,
            )
        self.database.update_job(job["id"], ocr_fingerprint=fingerprint)
        with tempfile.TemporaryDirectory(prefix="paperless-clerk-") as temp_directory:
            path = Path(temp_directory) / "document"
            source_hash = await paperless.download_document(int(document["id"]), path)
            if job.get("source_hash") and job["source_hash"] != source_hash:
                self.database.clear_pages(job["id"])
                self.database.add_event(
                    job["id"],
                    "warning",
                    "source_changed",
                    "Source changed; saved page results were discarded",
                )
            self.database.update_job(job["id"], source_hash=source_hash)

            with DocumentRenderer(
                path,
                dpi=self.settings.render_dpi,
                max_pixels=self.settings.max_image_pixels,
                jpeg_quality=self.settings.jpeg_quality,
                image_format=image_format,
            ) as renderer:
                total = renderer.page_count
                if total < 1:
                    raise ProcessingError("empty_document", "Document contains no renderable pages")
                self.database.update_job(job["id"], phase="ocr", current=0, total=total)
                await self._ocr_pages(job, renderer, model)

        results = self.database.page_results(job["id"])
        failures = [page for page in results if page["status"] != "complete"]
        if failures:
            pages = ", ".join(str(page["page_number"]) for page in failures[:20])
            retryable = any(page["status"] == "failed_retryable" for page in failures)
            raise ProcessingError(
                "page_ocr_failed",
                f"OCR did not complete for {len(failures)} page(s): {pages}",
                retryable=retryable,
            )

        pages = [(int(page["page_number"]), str(page["text"] or "")) for page in results]
        generated = assemble_pages(pages)
        if not meaningful_ocr(generated, self.settings.ocr_min_chars):
            raise ProcessingError(
                "ocr_not_meaningful",
                "The complete OCR result did not contain enough meaningful text to publish",
            )
        # OCR can take a long time. Re-read Paperless before any decision so a
        # user correction made while the model was running is never replaced
        # or compared against a stale snapshot.
        live_document = await paperless.get_document(int(document["id"]))
        if live_document.get("modified") != document.get("modified"):
            with tempfile.TemporaryDirectory(prefix="paperless-clerk-recheck-") as directory:
                live_hash = await paperless.download_document(
                    int(document["id"]), Path(directory) / "document"
                )
            if live_hash != source_hash:
                raise ProcessingError(
                    "source_changed",
                    "The Paperless source file changed during OCR; retrying the new version",
                    retryable=True,
                )
        document = live_document
        existing = str(document.get("content") or "")
        has_paperless_ocr = meaningful_ocr(existing, self.settings.ocr_min_chars)
        if not has_paperless_ocr and not deepseek_vllm_profile:
            self.database.update_job(job["id"], phase="publishing_ocr")
            updated = await paperless.update_document(int(document["id"]), {"content": generated})
            self.database.add_event(
                job["id"],
                "info",
                "ocr_published",
                f"Published complete OCR from {len(pages)} page(s)",
                {"pages": len(pages)},
            )
            if job["mode"] == "ocr":
                return (
                    ProcessOutcome("completed", "complete"),
                    generated,
                    pages,
                    updated,
                    source_hash,
                )
            return None, generated, pages, updated, source_hash

        self.database.update_job(job["id"], phase="verifying_ocr")
        comparison = compare_ocr(existing, generated, self.settings.ocr_similarity_threshold)
        # Similarity is symmetric, but replacement safety is not. If the new
        # result omits a meaningful amount of text already present in
        # Paperless, retain the system-of-record copy even when the overall
        # document remains similar. Both OCR versions remain available in job
        # diagnostics; no generated text is patched or heuristically repaired.
        coverage_safeguard = bool(
            comparison.is_similar
            and self.settings.prefer_clerk_ocr
            and comparison.existing_coverage < 0.98
            and (
                comparison.existing_missing_tokens >= 5
                or comparison.unmatched_existing_suffix_tokens >= 5
            )
        )
        guarded_review = deepseek_vllm_profile and not has_paperless_ocr
        allow_generated_publication = self.settings.prefer_clerk_ocr and not deepseek_vllm_profile
        selected_source = (
            "manual_review"
            if guarded_review
            else "clerk"
            if comparison.is_similar
            and allow_generated_publication
            and not coverage_safeguard
            else "paperless"
            if comparison.is_similar
            else "manual_review"
        )
        self.database.add_event(
            job["id"],
            (
                "warning"
                if guarded_review or coverage_safeguard or not comparison.is_similar
                else "info"
            ),
            "ocr_compared",
            f"OCR comparison score {comparison.score:.3f}; selected {selected_source.replace('_', ' ')}",
            {
                **comparison.metrics(),
                "selected_source": selected_source,
                "coverage_safeguard": coverage_safeguard,
                "guarded_review": guarded_review,
            },
        )
        if comparison.is_similar and not guarded_review:
            selected_text = existing
            selected_pages = pages_from_assembled_text(existing)
            if coverage_safeguard:
                self.database.add_event(
                    job["id"],
                    "warning",
                    "ocr_coverage_safeguard",
                    "Retained Paperless OCR because Clerk omitted material existing text",
                    {
                        "existing_missing_tokens": comparison.existing_missing_tokens,
                        "unmatched_existing_suffix_tokens": (
                            comparison.unmatched_existing_suffix_tokens
                        ),
                        "existing_coverage": comparison.existing_coverage,
                    },
                )
            elif deepseek_vllm_profile:
                self.database.add_event(
                    job["id"],
                    "info",
                    "ocr_guarded_retained_paperless",
                    "Retained existing Paperless OCR; guarded DeepSeek output is advisory",
                    {"profile": self.settings.ocr_profile, "score": comparison.score},
                )
            elif allow_generated_publication:
                selected_text = generated
                selected_pages = pages
                if generated != existing:
                    self.database.update_job(job["id"], phase="publishing_ocr")
                    document = await paperless.update_document(
                        int(document["id"]), {"content": generated}
                    )
                    self.database.add_event(
                        job["id"],
                        "info",
                        "ocr_preference_applied",
                        "Replaced existing Paperless OCR with Clerk OCR after a trusted match",
                        {"score": comparison.score, "selected_source": "clerk"},
                    )
            if job["mode"] == "ocr":
                return (
                    ProcessOutcome("completed", "complete"),
                    selected_text,
                    selected_pages,
                    document,
                    source_hash,
                )
            return None, selected_text, selected_pages, document, source_hash

        if guarded_review:
            self.database.add_event(
                job["id"],
                "warning",
                "ocr_guarded_review",
                "DeepSeek OCR requires review because Paperless has no existing OCR baseline",
                {
                    "profile": self.settings.ocr_profile,
                    "reason": "model_required_content_changing_repetition_guard",
                },
            )

        conflict_tag = await paperless.ensure_tag(self.settings.conflict_tag)
        tag_ids = {int(value) for value in document.get("tags", [])}
        tag_ids.add(int(conflict_tag["id"]))
        updated = await paperless.update_document(int(document["id"]), {"tags": sorted(tag_ids)})
        self.database.create_conflict(
            job_id=job["id"],
            document_id=int(document["id"]),
            document_title=str(document.get("title") or ""),
            existing_text=existing,
            generated_text=generated,
            score=comparison.score,
            metrics={**comparison.metrics(), "guarded_review": guarded_review},
            diff=comparison.mismatch_snippets,
            tag_id=int(conflict_tag["id"]),
        )
        return (
            ProcessOutcome(
                "needs_review",
                "ocr_conflict",
                (
                    "Guarded DeepSeek OCR has no existing Paperless baseline; review is "
                    "required before metadata analysis"
                    if guarded_review
                    else "Existing and Clerk OCR differ; review is required before metadata analysis"
                ),
            ),
            existing,
            pages,
            updated,
            source_hash,
        )

    async def _ocr_pages(
        self,
        job: dict[str, Any],
        renderer: DocumentRenderer,
        model: OpenAICompatibleClient,
    ) -> None:
        existing = {page["page_number"]: page for page in self.database.page_results(job["id"])}
        completed = sum(1 for page in existing.values() if page["status"] == "complete")
        self.database.update_job(job["id"], current=completed, total=renderer.page_count)
        pending: set[asyncio.Task[None]] = set()
        progress_lock = asyncio.Lock()

        async def process_page(page_number: int, image: bytes) -> None:
            nonlocal completed
            prior_attempts = int(existing.get(page_number, {}).get("attempts") or 0)
            started = time.monotonic()
            try:
                prompt = (
                    DEEPSEEK_OCR_PAGE_PROMPT
                    if self.settings.ocr_profile == "deepseek_ocr"
                    else DEEPSEEK_FREE_OCR_PAGE_PROMPT
                    if self.settings.ocr_profile == "deepseek_ocr_llamacpp"
                    else OCR_PAGE_PROMPT
                )
                text = await model.ocr_page(image, page_number=page_number, prompt=prompt)
                duration = int((time.monotonic() - started) * 1000)
                self.database.upsert_page(
                    job["id"],
                    page_number,
                    "complete",
                    prior_attempts + 1,
                    text=text,
                    duration_ms=duration,
                )
                async with progress_lock:
                    completed += 1
                    self.database.update_job(
                        job["id"], current=completed, total=renderer.page_count
                    )
            except ModelError as exc:
                duration = int((time.monotonic() - started) * 1000)
                status = "failed_retryable" if exc.retryable else "failed"
                self.database.upsert_page(
                    job["id"],
                    page_number,
                    status,
                    prior_attempts + 1,
                    error=str(exc)[:1500],
                    duration_ms=duration,
                )

        try:
            for page_index in range(renderer.page_count):
                page_number = page_index + 1
                if existing.get(page_number, {}).get("status") == "complete":
                    continue
                while len(pending) >= self.settings.page_concurrency:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    await asyncio.gather(*done)
                try:
                    # Tests and alternate renderers may provide a native async
                    # implementation. PyMuPDF rendering is moved off the API event
                    # loop in production.
                    render_async = getattr(renderer, "render_async", None)
                    image = (
                        await render_async(page_index)
                        if render_async is not None
                        else await asyncio.to_thread(renderer.render, page_index)
                    )
                except RenderError as exc:
                    prior_attempts = int(existing.get(page_number, {}).get("attempts") or 0)
                    self.database.upsert_page(
                        job["id"],
                        page_number,
                        "failed",
                        prior_attempts + 1,
                        error=str(exc)[:1500],
                    )
                    continue
                pending.add(asyncio.create_task(process_page(page_number, image)))
            if pending:
                await asyncio.gather(*pending)
        except BaseException:
            # A shutdown or unexpected sibling failure must not leave model
            # requests detached from their owning job.
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise

    async def _process_metadata(
        self,
        job: dict[str, Any],
        document: dict[str, Any],
        pages: list[tuple[int, str]],
        paperless: PaperlessClient,
        model: OpenAICompatibleClient,
    ) -> dict[str, Any]:
        self.database.update_job(job["id"], phase="loading_vocabulary", current=0, total=0)
        catalog = await paperless.catalog()
        model_document, model_catalog, _ = _without_watch_tag(
            document, catalog, self.settings.automation_tag
        )
        analyzer = MetadataAnalyzer(self.settings, model)

        def progress(phase: str, current: int, total: int) -> None:
            self.database.update_job(job["id"], phase=phase, current=current, total=total)

        proposal, trace = await analyzer.analyze(
            document=model_document, pages=pages, catalog=model_catalog, progress=progress
        )
        self.database.update_job(job["id"], phase="validating_metadata", current=0, total=0)
        # Model inference runs concurrently, but vocabulary validation and
        # creation are serialized. Re-fetching inside the lock means two jobs
        # proposing aliases cannot create near-duplicates from stale catalogs.
        async with self.vocabulary_lock:
            live_document = await paperless.get_document(int(document["id"]))
            if str(live_document.get("content") or "") != str(document.get("content") or ""):
                raise ProcessingError(
                    "document_changed",
                    "Paperless OCR changed during metadata analysis; retrying with the new text",
                    retryable=True,
                )
            live_catalog = await paperless.catalog()
            _, planning_catalog, watch_tag = _without_watch_tag(
                live_document, live_catalog, self.settings.automation_tag
            )
            plan = MetadataPlanner(self.settings).validate(
                proposal, planning_catalog, allowed_ids=trace["candidate_ids"]
            )
            existing_tag_ids = {int(tag_id) for tag_id in live_document.get("tags", [])}
            if watch_tag:
                existing_tag_ids.discard(int(watch_tag["id"]))
            tagless = not existing_tag_ids and not plan.tags
            tag_review = {
                **trace.get(
                    "tag_review",
                    {
                        "performed": False,
                        "status": "unknown",
                        "assessment": "No focused tag review information was recorded.",
                        "selected_count": len(proposal.tags),
                    },
                ),
                "accepted_count": len(plan.tags),
                "existing_count": len(existing_tag_ids),
                "outcome": "no_tags" if tagless else "tags_available",
            }
            before = {
                key: live_document.get(key)
                for key in (
                    "title",
                    "correspondent",
                    "document_type",
                    "tags",
                    "created",
                    "custom_fields",
                    "modified",
                )
            }
            self.database.update_job(job["id"], phase="applying_metadata")
            applied, updated = await apply_metadata_plan(
                plan=plan,
                document=live_document,
                paperless=paperless,
                settings=self.settings,
                consume_tag=watch_tag,
            )
        rationale = {
            "summary": proposal.summary,
            "rejected": plan.rejected,
            "candidate_duplicates": plan.duplicate_candidates,
            "source_chunks": trace["chunks"],
            "candidate_counts": trace["candidate_counts"],
            "candidate_ids": trace["candidate_ids"],
            "candidate_vocabulary": trace.get("candidate_vocabulary", {}),
            "tag_review": tag_review,
            "model_diagnostics": trace.get("model_diagnostics", []),
        }
        decision_status = "no_tags" if tagless else ("applied" if applied["patch"] else "no_change")
        decision_id = self.database.add_decision(
            job_id=job["id"],
            document_id=int(document["id"]),
            document_title=str(updated.get("title") or document.get("title") or ""),
            proposal=proposal.model_dump(mode="json"),
            applied=applied,
            rationale=rationale,
            before=before,
            status=decision_status,
        )
        log.info(
            "Metadata decision recorded job=%s document=%s decision=%s status=%s "
            "tags=%s reused=%s created=%s rejected=%s diagnostics=%s",
            job["id"],
            document["id"],
            decision_id,
            decision_status,
            len(plan.tags),
            len(applied["reused"]),
            len(applied["created"]),
            len(plan.rejected),
            len(rationale["model_diagnostics"]),
        )
        if tag_review["performed"]:
            audit_suffix = (
                " after challenging an empty first result"
                if tag_review.get("abstention_audit_performed")
                else ""
            )
            self.database.add_event(
                job["id"],
                "info" if tag_review["accepted_count"] else "warning",
                "metadata_tag_review",
                (
                    f"Focused tag review accepted {tag_review['accepted_count']} tag(s)"
                    f"{audit_suffix}"
                    if tag_review["accepted_count"]
                    else f"Focused tag review did not yield an applicable tag{audit_suffix}"
                ),
                {
                    "decision_id": decision_id,
                    "status": tag_review["status"],
                    "accepted_count": tag_review["accepted_count"],
                    "abstention_audit_status": tag_review.get("abstention_audit_status"),
                },
            )
        self.database.add_event(
            job["id"],
            "info",
            "metadata_decision",
            "Metadata decision applied"
            if applied["patch"]
            else "Metadata analysis required no changes",
            {"decision_id": decision_id, "changed_fields": sorted(applied["patch"])},
        )
        if tagless:
            self.database.add_event(
                job["id"],
                "warning",
                "metadata_no_tags",
                "Metadata completed without a document tag; review the focused assessment",
                {"decision_id": decision_id},
            )
        if applied["removed"]:
            self.database.add_event(
                job["id"],
                "info",
                "watch_tag_removed",
                f"Removed automation watch tag {applied['removed'][0]['name']}",
                {"tag_id": applied["removed"][0]["id"]},
            )
        return updated


class JobManager:
    def __init__(self, database: Database, settings_manager: SettingsManager):
        self.database = database
        self.settings_manager = settings_manager
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._discovery_wake = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._discovery_task: asyncio.Task[None] | None = None
        self._vocabulary_lock = asyncio.Lock()
        self._discovery_page = 2
        self._scan_backlog_next = False
        self._discovery_signature: tuple[str, str, int] | None = None
        self._discovery_failure_active = False

    async def start(self) -> None:
        settings = self.settings_manager.get()
        self._tasks = [
            asyncio.create_task(self._worker(index), name=f"clerk-worker-{index}")
            for index in range(settings.job_workers)
        ]
        self._discovery_task = asyncio.create_task(self._discover_loop(), name="clerk-discovery")

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._discovery_wake.set()
        tasks = [*self._tasks, *([self._discovery_task] if self._discovery_task else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def wake(self) -> None:
        self._wake.set()

    def settings_changed(self) -> None:
        self._discovery_wake.set()

    async def _worker(self, index: int) -> None:
        worker_id = f"{uuid.uuid4()}:{index}"
        while not self._stop.is_set():
            settings = self.settings_manager.get()
            job = self.database.claim_job(worker_id, settings.lease_seconds)
            if not job:
                self._wake.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                continue
            self.database.add_event(
                job["id"],
                "info",
                "started",
                f"Processing attempt {job['attempt']} of {job['max_attempts']}",
            )
            log.info(
                "Processing job started job=%s document=%s mode=%s attempt=%s/%s",
                job["id"],
                job["document_id"],
                job["mode"],
                job["attempt"],
                job["max_attempts"],
            )
            heartbeat = asyncio.create_task(self._heartbeat(job["id"], settings.lease_seconds))
            try:
                outcome = await DocumentProcessor(
                    self.database, settings, self._vocabulary_lock
                ).process(job)
                if outcome.status == "needs_review":
                    self.database.mark_needs_review(job["id"], outcome.phase, outcome.message)
                    self.database.clear_pages(job["id"])
                    log.warning(
                        "Processing job needs review job=%s document=%s phase=%s: %s",
                        job["id"],
                        job["document_id"],
                        outcome.phase,
                        outcome.message,
                    )
                    await self._notify_job_issue(
                        job,
                        kind=outcome.phase,
                        message=outcome.message or "Document processing requires review",
                    )
                else:
                    self.database.finish_job(job["id"], outcome.status, outcome.phase)
                    self.database.clear_pages(job["id"])
                    self.database.add_event(
                        job["id"], "info", "completed", "Document processing completed"
                    )
                    log.info(
                        "Processing job completed job=%s document=%s status=%s phase=%s",
                        job["id"],
                        job["document_id"],
                        outcome.status,
                        outcome.phase,
                    )
            except ProcessingError as exc:
                status = self.database.fail_or_retry(job["id"], exc.code, str(exc), exc.retryable)
                log_method = log.error if status == "failed" else log.warning
                log_method(
                    "Processing job %s job=%s document=%s code=%s retryable=%s: %s",
                    status,
                    job["id"],
                    job["document_id"],
                    exc.code,
                    exc.retryable,
                    str(exc),
                )
                if status == "failed":
                    self.database.update_document_state_status(int(job["document_id"]), "failed")
                    await self._notify_job_issue(job, kind="failed", message=str(exc))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - final process boundary
                log.exception("Unhandled processing error for job %s", job["id"])
                status = self.database.fail_or_retry(job["id"], "internal_error", str(exc), False)
                self.database.update_document_state_status(int(job["document_id"]), "failed")
                if status == "failed":
                    await self._notify_job_issue(job, kind="failed", message=str(exc))
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _notify_job_issue(self, job: dict[str, Any], *, kind: str, message: str) -> None:
        current = self.database.get_job(job["id"]) or job
        document_id = int(current["document_id"])
        document_title = str(current.get("document_title") or f"Document {document_id}")
        if kind == "ocr_conflict":
            title = "Paperless Clerk: OCR conflict"
            tags = ("warning", "mag")
        elif kind == "failed":
            title = "Paperless Clerk: processing failed"
            tags = ("x", "page_facing_up")
        else:
            title = "Paperless Clerk: review required"
            tags = ("warning", "page_facing_up")
        await self._publish_notification(
            kind=kind,
            title=title,
            message=f"{document_title} (document #{document_id})\n{message}",
            tags=tags,
            job_id=str(job["id"]),
        )

    async def _publish_notification(
        self,
        *,
        kind: str,
        title: str,
        message: str,
        tags: tuple[str, ...],
        job_id: str | None = None,
    ) -> None:
        settings = self.settings_manager.get()
        if not settings.notifications_enabled:
            return
        client: NtfyClient | None = None
        try:
            client = NtfyClient(settings)
            await client.publish(title=title, message=message, priority=4, tags=tags)
            log.info("ntfy notification sent kind=%s job=%s", kind, job_id or "-")
            if job_id:
                self.database.add_event(
                    job_id,
                    "info",
                    "notification_sent",
                    "ntfy notification sent",
                    {"kind": kind},
                )
        except Exception as exc:  # Notification delivery must not change a job outcome.
            log.warning(
                "ntfy notification failed kind=%s job=%s: %s",
                kind,
                job_id or "-",
                str(exc),
            )
            if job_id:
                self.database.add_event(
                    job_id,
                    "warning",
                    "notification_failed",
                    f"ntfy notification failed: {str(exc)[:800]}",
                    {"kind": kind},
                )
        finally:
            if client:
                with contextlib.suppress(Exception):
                    await client.close()

    async def _heartbeat(self, job_id: str, lease_seconds: int) -> None:
        while True:
            await asyncio.sleep(max(20, lease_seconds // 3))
            self.database.heartbeat(job_id, lease_seconds)

    async def _discover_loop(self) -> None:
        while not self._stop.is_set():
            self._discovery_wake.clear()
            settings = self.settings_manager.get()
            if settings.automation_enabled and settings.paperless_token.get_secret_value():
                paperless = PaperlessClient(settings)
                try:
                    signature = (
                        settings.paperless_url,
                        settings.automation_tag,
                        settings.automation_page_size,
                    )
                    if signature != self._discovery_signature:
                        self._discovery_page = 2
                        self._scan_backlog_next = False
                        self._discovery_signature = signature
                    page_to_scan = self._discovery_page if self._scan_backlog_next else 1
                    documents, has_next = await paperless.list_documents(
                        page_size=settings.automation_page_size,
                        page=page_to_scan,
                        tag_name=settings.automation_tag,
                    )
                    if page_to_scan == 1:
                        self._scan_backlog_next = has_next
                        if not has_next:
                            self._discovery_page = 2
                    else:
                        self._discovery_page = page_to_scan + 1 if has_next else 2
                        self._scan_backlog_next = False
                    states = self.database.document_states(int(item["id"]) for item in documents)
                    for document in documents:
                        document_id = int(document["id"])
                        state = states.get(document_id)
                        if state and state.get("paperless_modified") == document.get("modified"):
                            continue
                        job, created = self.database.enqueue_job(
                            document_id,
                            "full",
                            settings.job_max_attempts,
                            document_title=str(document.get("title") or ""),
                        )
                        if created:
                            # Record discovery immediately. If Paperless later
                            # rejects the fetch, the unchanged document will not
                            # generate a brand-new failed job every poll cycle.
                            self.database.set_document_state(
                                document_id,
                                document.get("modified"),
                                None,
                                job["id"],
                                "queued",
                            )
                            self.wake()
                    self._discovery_failure_active = False
                except Exception as exc:
                    # A shrinking result set can invalidate a later page. The
                    # next poll restarts safely from the newest documents.
                    self._discovery_page = 2
                    self._scan_backlog_next = False
                    log.warning("Automatic Paperless discovery failed", exc_info=True)
                    if not self._discovery_failure_active:
                        await self._publish_notification(
                            kind="discovery_failed",
                            title="Paperless Clerk: discovery failed",
                            message=f"Automatic Paperless discovery failed: {str(exc)[:1000]}",
                            tags=("warning", "satellite"),
                        )
                        self._discovery_failure_active = (
                            self.settings_manager.get().notifications_enabled
                        )
                finally:
                    await paperless.close()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._discovery_wake.wait(), timeout=settings.automation_interval_seconds
                )


async def resolve_conflict(
    *,
    database: Database,
    settings: Settings,
    conflict_id: str,
    resolution: str,
) -> dict[str, Any]:
    conflict = database.get_conflict(conflict_id)
    if not conflict or conflict["status"] != "open":
        raise ProcessingError("conflict_not_found", "Open OCR conflict was not found")
    if not database.claim_conflict_resolution(conflict_id, resolution):
        raise ProcessingError(
            "conflict_changed", "OCR conflict is already being resolved by another request"
        )
    paperless = PaperlessClient(settings)
    try:
        document = await paperless.get_document(int(conflict["document_id"]))
        live_content = str(document.get("content") or "")
        if resolution == "keep_existing" and not meaningful_ocr(
            live_content, settings.ocr_min_chars
        ):
            raise ProcessingError(
                "no_existing_ocr",
                "Paperless still has no meaningful OCR to keep. Add or correct OCR in "
                "Paperless, then retry this choice, or explicitly choose Clerk OCR.",
            )
        if resolution == "use_clerk" and live_content != str(conflict["existing_text"] or ""):
            raise ProcessingError(
                "conflict_source_changed",
                "Paperless OCR changed after this review was created. Clerk did not overwrite "
                "the newer text; keep the current Paperless OCR or start a new comparison.",
            )
        patch: dict[str, Any] = {}
        if resolution == "use_clerk":
            patch["content"] = conflict["generated_text"]
        tag_id = conflict.get("conflict_tag_id")
        if tag_id:
            patch["tags"] = [
                int(item) for item in document.get("tags", []) if int(item) != int(tag_id)
            ]
        updated = await paperless.update_document(int(document["id"]), patch) if patch else document
        if not database.resolve_conflict(conflict_id, resolution):
            raise ProcessingError("conflict_changed", "OCR conflict was resolved concurrently")
        database.finish_job(conflict["job_id"], "completed", f"ocr_conflict_{resolution}")
        database.add_event(
            conflict["job_id"],
            "info",
            "conflict_resolved",
            f"OCR conflict resolved with {resolution.replace('_', ' ')}",
        )
        job, _ = database.enqueue_job(
            int(document["id"]),
            "metadata",
            settings.job_max_attempts,
            document_title=str(updated.get("title") or document.get("title") or ""),
        )
        database.set_document_state(
            int(document["id"]), updated.get("modified"), None, job["id"], "queued"
        )
        return {"conflict_id": conflict_id, "resolution": resolution, "job": job}
    except PaperlessError as exc:
        database.release_conflict_resolution(conflict_id, resolution)
        raise ProcessingError("paperless_error", str(exc), retryable=exc.retryable) from exc
    except Exception:
        database.release_conflict_resolution(conflict_id, resolution)
        raise
    finally:
        await paperless.close()
