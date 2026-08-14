from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from paperless_clerk.clients.openai_compatible import ModelError, OpenAICompatibleClient
from paperless_clerk.clients.paperless import PaperlessClient
from paperless_clerk.config import Settings
from paperless_clerk.domain.chunking import TextChunk, chunk_pages
from paperless_clerk.domain.taxonomy import (
    Entity,
    duplicate_similarity,
    entities_from_payload,
    find_duplicate,
    name_tokens,
    select_candidates,
)
from paperless_clerk.prompts import (
    METADATA_REDUCE_SYSTEM_PROMPT,
    METADATA_SYSTEM_PROMPT,
    TAG_ABSTENTION_AUDIT_SYSTEM_PROMPT,
    TAG_REVIEW_SYSTEM_PROMPT,
)
from paperless_clerk.schemas import (
    ChunkAnalysis,
    CustomFieldCandidate,
    MetadataChoice,
    MetadataProposal,
    NewCustomFieldCandidate,
    ScalarCandidate,
    TagReview,
)

ProgressCallback = Callable[[str, int, int], None]
log = logging.getLogger(__name__)

MAX_MODEL_DIAGNOSTICS = 50
MODEL_OUTPUT_PREVIEW_CHARS = 2_500


def _validation_details(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {
            "path": ".".join(str(part) for part in error["loc"]),
            "type": str(error["type"]),
            "message": str(error["msg"]),
        }
        for error in exc.errors(include_url=False, include_input=False)[:12]
    ]


def _output_preview(value: Any) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = repr(value)
    return rendered[:MODEL_OUTPUT_PREVIEW_CHARS]


@dataclass(frozen=True)
class CandidateCatalog:
    tags: list[Entity]
    correspondents: list[Entity]
    document_types: list[Entity]
    custom_fields: list[dict[str, Any]]

    def limited(self, limit: int) -> CandidateCatalog:
        return CandidateCatalog(
            tags=self.tags[:limit],
            correspondents=self.correspondents[:limit],
            document_types=self.document_types[:limit],
            custom_fields=self.custom_fields[:limit],
        )

    def prompt_dict(self) -> dict[str, Any]:
        return {
            "tags": [item.prompt_dict() for item in self.tags],
            "correspondents": [item.prompt_dict() for item in self.correspondents],
            "document_types": [item.prompt_dict() for item in self.document_types],
            "custom_fields": [
                {
                    "id": item.get("id"),
                    "name": str(item.get("name", ""))[:128],
                    "data_type": item.get("data_type"),
                    "select_options": [
                        {"id": option.get("id"), "label": str(option.get("label", ""))[:128]}
                        for option in (item.get("extra_data") or {}).get("select_options", [])[:50]
                    ],
                }
                for item in self.custom_fields
            ],
        }


class MetadataAnalyzer:
    def __init__(self, settings: Settings, model: OpenAICompatibleClient):
        self.settings = settings
        self.model = model
        self._document_id: int | None = None
        self._model_diagnostics: list[dict[str, Any]] = []

    def _record_diagnostic(self, entry: dict[str, Any]) -> None:
        if len(self._model_diagnostics) < MAX_MODEL_DIAGNOSTICS:
            self._model_diagnostics.append(entry)
            return
        if self._model_diagnostics[-1].get("stage") == "diagnostic_limit":
            self._model_diagnostics[-1]["omitted_entries"] += 1
            return
        self._model_diagnostics[-1] = {
            "stage": "diagnostic_limit",
            "status": "truncated",
            "omitted_entries": 1,
        }

    def _record_validation_failure(
        self,
        *,
        stage: str,
        attempt: int,
        exc: ValidationError,
        raw: Any,
        scope: dict[str, Any] | None = None,
    ) -> None:
        errors = _validation_details(exc)
        self._record_diagnostic(
            {
                "stage": stage,
                "status": "validation_error",
                "attempt": attempt + 1,
                "maximum_attempts": self.settings.model_max_retries + 1,
                **(scope or {}),
                "errors": errors,
                "output_preview": _output_preview(raw),
            }
        )
        summary = "; ".join(f"{error['path'] or '<root>'}: {error['message']}" for error in errors)
        log.warning(
            "Metadata output validation failed document=%s stage=%s attempt=%s/%s: %s",
            self._document_id,
            stage,
            attempt + 1,
            self.settings.model_max_retries + 1,
            summary[:800],
        )

    def _record_model_failure(
        self,
        *,
        stage: str,
        attempt: int,
        exc: ModelError,
        scope: dict[str, Any] | None = None,
    ) -> None:
        self._record_diagnostic(
            {
                "stage": stage,
                "status": "model_error",
                "attempt": attempt + 1,
                "maximum_attempts": self.settings.model_max_retries + 1,
                **(scope or {}),
                "retryable": exc.retryable,
                "error": str(exc)[:1_000],
            }
        )
        log.warning(
            "Metadata model call failed document=%s stage=%s attempt=%s/%s retryable=%s: %s",
            self._document_id,
            stage,
            attempt + 1,
            self.settings.model_max_retries + 1,
            exc.retryable,
            str(exc)[:800],
        )

    def candidates(
        self,
        text: str,
        document: dict[str, Any],
        catalog: dict[str, list[dict[str, Any]]],
    ) -> CandidateCatalog:
        limit = self.settings.metadata_candidate_limit
        tags = entities_from_payload(catalog["tags"])
        correspondents = entities_from_payload(catalog["correspondents"])
        document_types = entities_from_payload(catalog["document_types"])
        custom_fields = entities_from_payload(catalog["custom_fields"])
        current_custom_fields = {
            int(value["field"])
            for value in document.get("custom_fields", [])
            if value.get("field") is not None
        }
        return CandidateCatalog(
            tags=select_candidates(
                text,
                tags,
                kind="tag",
                current_ids={int(item) for item in document.get("tags", [])},
                limit=limit,
            ),
            correspondents=select_candidates(
                text,
                correspondents,
                kind="correspondent",
                current_ids={int(document["correspondent"])}
                if document.get("correspondent")
                else set(),
                limit=limit,
            ),
            document_types=select_candidates(
                text,
                document_types,
                kind="document_type",
                current_ids={int(document["document_type"])}
                if document.get("document_type")
                else set(),
                limit=limit,
            ),
            custom_fields=[
                item.raw or {}
                for item in select_candidates(
                    text,
                    custom_fields,
                    kind="custom_field",
                    current_ids=current_custom_fields,
                    limit=limit,
                )
            ],
        )

    async def analyze(
        self,
        *,
        document: dict[str, Any],
        pages: list[tuple[int, str]],
        catalog: dict[str, list[dict[str, Any]]],
        progress: ProgressCallback | None = None,
    ) -> tuple[MetadataProposal, dict[str, Any]]:
        self._document_id = int(document["id"]) if document.get("id") is not None else None
        self._model_diagnostics = []
        full_text = "\n\n".join(text for _, text in pages)
        candidates = self.candidates(full_text, document, catalog)
        input_budget = max(
            4_000,
            (
                self.settings.metadata_context_tokens
                - self.settings.metadata_max_output_tokens
                - 1_000
            )
            * 3,
        )
        candidate_limit = self.settings.metadata_candidate_limit
        while candidate_limit > 5:
            candidate_size = len(json.dumps(candidates.prompt_dict(), ensure_ascii=False))
            if candidate_size <= input_budget * 0.45:
                break
            candidate_limit = max(5, candidate_limit // 2)
            candidates = candidates.limited(candidate_limit)
        candidate_size = len(json.dumps(candidates.prompt_dict(), ensure_ascii=False))
        available_text = input_budget - candidate_size - 6_000
        if available_text < 1_000:
            raise ModelError(
                "Metadata context limit is too small for the current controlled vocabulary; "
                "increase the context limit or lower the candidate limit"
            )
        chunks = chunk_pages(pages, min(self.settings.metadata_chunk_chars, available_text))
        if not chunks:
            raise ModelError("Document contains no text to classify")
        log.info(
            "Metadata analysis prepared document=%s pages=%s chunks=%s candidates="
            "tags:%s correspondents:%s document_types:%s custom_fields:%s new_tags_allowed=%s",
            self._document_id,
            len(pages),
            len(chunks),
            len(candidates.tags),
            len(candidates.correspondents),
            len(candidates.document_types),
            len(candidates.custom_fields),
            self.settings.allow_new_tags,
        )

        semaphore = asyncio.Semaphore(self.settings.metadata_concurrency)
        completed = 0
        completed_lock = asyncio.Lock()

        async def map_one(chunk: TextChunk) -> ChunkAnalysis:
            nonlocal completed
            async with semaphore:
                result = await self._map_chunk(document, chunk, candidates)
            async with completed_lock:
                completed += 1
                if progress:
                    progress("metadata_map", completed, len(chunks))
            return result

        map_tasks = [asyncio.create_task(map_one(chunk)) for chunk in chunks]
        try:
            mapped = await asyncio.gather(*map_tasks)
        except Exception:
            for task in map_tasks:
                task.cancel()
            await asyncio.gather(*map_tasks, return_exceptions=True)
            raise
        proposal = await self._hierarchical_reduce(document, mapped, candidates, progress)
        tag_review = {
            "performed": False,
            "status": "not_needed",
            "assessment": "The general metadata pass produced at least one usable tag candidate.",
            "selected_count": len(proposal.tags),
            "new_tags_allowed": self.settings.allow_new_tags,
        }
        if not document.get("tags") and not self._has_usable_tag(proposal, candidates):
            tag_review["performed"] = True
            if progress:
                progress("metadata_tag_review", 0, 1)
            try:
                reviewed, review_details = await self._review_tags(
                    document, pages, candidates, proposal
                )
                proposal = proposal.model_copy(update={"tags": reviewed.tags})
                tag_review.update(
                    {
                        "status": "selected" if reviewed.tags else "abstained",
                        "assessment": reviewed.assessment,
                        "selected_count": len(reviewed.tags),
                        **review_details,
                    }
                )
            except ModelError as exc:
                tag_review.update(
                    {
                        "status": "failed",
                        "assessment": f"Focused tag review could not complete: {exc}",
                        "selected_count": 0,
                    }
                )
            if progress:
                progress("metadata_tag_review", 1, 1)
        trace = {
            "chunks": [
                {"index": chunk.index, "page_start": chunk.page_start, "page_end": chunk.page_end}
                for chunk in chunks
            ],
            "candidate_counts": {
                "tags": len(candidates.tags),
                "correspondents": len(candidates.correspondents),
                "document_types": len(candidates.document_types),
                "custom_fields": len(candidates.custom_fields),
            },
            "candidate_ids": {
                "tag": [item.id for item in candidates.tags],
                "correspondent": [item.id for item in candidates.correspondents],
                "document_type": [item.id for item in candidates.document_types],
                "custom_field": [int(item["id"]) for item in candidates.custom_fields],
            },
            "candidate_vocabulary": candidates.prompt_dict(),
            "tag_review": tag_review,
            "model_diagnostics": self._model_diagnostics,
        }
        return proposal, trace

    def _has_usable_tag(self, proposal: MetadataProposal, candidates: CandidateCatalog) -> bool:
        candidate_ids = {item.id for item in candidates.tags}
        watch_name = " ".join(self.settings.automation_tag.split()).casefold()
        for tag in proposal.tags:
            if tag.confidence < self.settings.metadata_min_confidence:
                continue
            if tag.existing_id is not None and tag.existing_id in candidate_ids:
                return True
            if (
                tag.new_name
                and self.settings.allow_new_tags
                and len(tag.reason.strip()) >= 12
                and name_tokens(tag.new_name, "tag")
                and " ".join(tag.new_name.split()).casefold() != watch_name
            ):
                return True
        return False

    async def _review_tags(
        self,
        document: dict[str, Any],
        pages: list[tuple[int, str]],
        candidates: CandidateCatalog,
        proposal: MetadataProposal,
    ) -> tuple[TagReview, dict[str, Any]]:
        candidate_tags = [item.prompt_dict() for item in candidates.tags]
        general_decision = {
            "correspondent": _compact_candidate(proposal.correspondent),
            "document_type": _compact_candidate(proposal.document_type),
            "tags": [_compact_candidate(item) for item in proposal.tags[:12]],
            "title": _compact_candidate(proposal.title),
            "document_date": _compact_candidate(proposal.document_date),
            "summary": proposal.summary[:700],
        }
        input_budget = max(
            4_000,
            (
                self.settings.metadata_context_tokens
                - self.settings.metadata_max_output_tokens
                - 1_000
            )
            * 3,
        )
        fixed_size = len(json.dumps(candidate_tags, ensure_ascii=False)) + len(
            json.dumps(general_decision, ensure_ascii=False)
        )
        excerpt_limit = max(1_000, min(18_000, input_budget - fixed_size - 5_000))
        payload = {
            "task": "Perform a focused second-pass tag decision.",
            "document": {
                "paperless_id": document.get("id"),
                "current_title": document.get("title", ""),
                "current_tag_ids": document.get("tags", []),
            },
            "controlled_vocabulary": {"tags": candidate_tags},
            "new_tags_allowed": self.settings.allow_new_tags,
            "decision_requirements": {
                "separate_form_from_subject": True,
                "prefer_broad_reusable_subject": True,
                "first_of_kind_can_create_tag": self.settings.allow_new_tags,
                "avoid_composite_document_type_tags": True,
            },
            "general_metadata_decision": general_decision,
            "representative_document_text": _representative_text(pages, excerpt_limit),
        }
        reviewed = await self._request_tag_review(
            payload,
            system=TAG_REVIEW_SYSTEM_PROMPT,
            stage="focused_tag_review",
            schema_name="clerk_tag_review",
            error_label="Focused tag review",
        )
        details: dict[str, Any] = {"abstention_audit_performed": False}
        if reviewed.tags or not self.settings.allow_new_tags:
            return reviewed, details

        log.info(
            "Focused tag review abstained; auditing controlled vocabulary growth document=%s",
            self._document_id,
        )
        audit_payload = {
            **payload,
            "task": "Audit and either overturn or confirm the prior empty tag decision.",
            "prior_focused_review": reviewed.model_dump(mode="json"),
        }
        details = {
            "abstention_audit_performed": True,
            "initial_assessment": reviewed.assessment,
        }
        try:
            audited = await self._request_tag_review(
                audit_payload,
                system=TAG_ABSTENTION_AUDIT_SYSTEM_PROMPT,
                stage="tag_abstention_audit",
                schema_name="clerk_tag_abstention_audit",
                error_label="Tag abstention audit",
            )
        except ModelError as exc:
            details.update(
                {
                    "abstention_audit_status": "failed",
                    "abstention_audit_error": str(exc)[:1_000],
                }
            )
            log.warning(
                "Tag abstention audit failed document=%s; retaining the initial abstention: %s",
                self._document_id,
                str(exc)[:800],
            )
            return reviewed, details
        details["abstention_audit_status"] = (
            "overturned" if audited.tags else "confirmed_abstention"
        )
        return audited, details

    async def _request_tag_review(
        self,
        payload: dict[str, Any],
        *,
        system: str,
        stage: str,
        schema_name: str,
        error_label: str,
    ) -> TagReview:
        user = json.dumps(payload, ensure_ascii=False)
        last_error: Exception | None = None
        for attempt in range(self.settings.model_max_retries + 1):
            try:
                raw = await self.model.structured(
                    name=schema_name,
                    schema=TagReview.model_json_schema(),
                    system=system,
                    user=user,
                )
                reviewed = TagReview.model_validate(raw)
                raw_assessment = raw.get("assessment")
                normalized_assessment = (
                    " ".join(raw_assessment.split()) if isinstance(raw_assessment, str) else ""
                )
                assessment_was_bounded = len(normalized_assessment) > len(reviewed.assessment)
                self._record_diagnostic(
                    {
                        "stage": stage,
                        "status": "normalized" if assessment_was_bounded else "completed",
                        "attempt": attempt + 1,
                        "maximum_attempts": self.settings.model_max_retries + 1,
                        "recovered_after_validation_error": attempt > 0,
                        "returned_tag_count": len(reviewed.tags),
                        "assessment_original_chars": len(normalized_assessment),
                        "assessment_stored_chars": len(reviewed.assessment),
                        "assessment": reviewed.assessment,
                    }
                )
                if assessment_was_bounded:
                    log.warning(
                        "%s normalized a verbose assessment document=%s characters=%s->%s; "
                        "retained %s tag choice(s)",
                        error_label,
                        self._document_id,
                        len(normalized_assessment),
                        len(reviewed.assessment),
                        len(reviewed.tags),
                    )
                log.info(
                    "%s completed document=%s tags=%s attempt=%s/%s",
                    error_label,
                    self._document_id,
                    len(reviewed.tags),
                    attempt + 1,
                    self.settings.model_max_retries + 1,
                )
                return reviewed
            except ValidationError as exc:
                last_error = exc
                self._record_validation_failure(stage=stage, attempt=attempt, exc=exc, raw=raw)
                previous = json.dumps(raw, ensure_ascii=False)[:4_000]
                errors = json.dumps(
                    exc.errors(include_url=False, include_input=False), ensure_ascii=False
                )[:2_000]
                user = (
                    json.dumps(payload, ensure_ascii=False)
                    + "\n\nRepair the previous output instead of repeating it. "
                    + "Use exactly the top-level keys tags and assessment."
                    + f"\nPrevious output: {previous}\nValidation errors: {errors}"
                )
            except ModelError as exc:
                self._record_model_failure(stage=stage, attempt=attempt, exc=exc)
                if not exc.retryable:
                    raise
                last_error = exc
            if attempt < self.settings.model_max_retries:
                await asyncio.sleep(min(2, 0.25 * (2**attempt)))
        raise ModelError(
            f"{error_label} remained invalid after retries: {last_error}", retryable=True
        ) from last_error

    async def _map_chunk(
        self, document: dict[str, Any], chunk: TextChunk, candidates: CandidateCatalog
    ) -> ChunkAnalysis:
        payload = {
            "task": "Extract candidate metadata from only this document chunk.",
            "document": {
                "paperless_id": document.get("id"),
                "current_title": document.get("title", ""),
                "current_correspondent_id": document.get("correspondent"),
                "current_document_type_id": document.get("document_type"),
                "current_tag_ids": document.get("tags", []),
                "current_custom_fields": document.get("custom_fields", []),
            },
            "controlled_vocabulary": candidates.prompt_dict(),
            "new_value_policy": {
                "tags": self.settings.allow_new_tags,
                "correspondents": self.settings.allow_new_correspondents,
                "document_types": self.settings.allow_new_document_types,
                "custom_field_definitions": self.settings.allow_new_custom_fields,
            },
            "source": {
                "chunk_index": chunk.index,
                "pages": [chunk.page_start, chunk.page_end],
                "text": chunk.text,
            },
        }
        user = json.dumps(payload, ensure_ascii=False)
        last_error: Exception | None = None
        for attempt in range(self.settings.model_max_retries + 1):
            try:
                raw = await self.model.structured(
                    name="clerk_chunk_analysis",
                    schema=ChunkAnalysis.model_json_schema(),
                    system=METADATA_SYSTEM_PROMPT,
                    user=user,
                )
                analysis = ChunkAnalysis.model_validate(raw)
                if attempt:
                    self._record_diagnostic(
                        {
                            "stage": "metadata_map",
                            "status": "recovered",
                            "attempt": attempt + 1,
                            "chunk_index": chunk.index,
                            "pages": [chunk.page_start, chunk.page_end],
                        }
                    )
                return analysis
            except ValidationError as exc:
                last_error = exc
                self._record_validation_failure(
                    stage="metadata_map",
                    attempt=attempt,
                    exc=exc,
                    raw=raw,
                    scope={
                        "chunk_index": chunk.index,
                        "pages": [chunk.page_start, chunk.page_end],
                    },
                )
                expected = ", ".join(ChunkAnalysis.model_fields)
                previous = json.dumps(raw, ensure_ascii=False)[:4_000]
                errors = json.dumps(
                    exc.errors(include_url=False, include_input=False), ensure_ascii=False
                )[:2_000]
                user = (
                    json.dumps(payload, ensure_ascii=False)
                    + "\n\nRepair the previous output instead of repeating it. "
                    + f"Use exactly these top-level keys: {expected}. "
                    + "Do not use flat title, date, correspondent_id, document_type_id, tag_ids, "
                    + "or custom_fields fields. Every non-empty candidate needs confidence and reason."
                    + f"\nPrevious output: {previous}\nValidation errors: {errors}"
                )
            except ModelError as exc:
                self._record_model_failure(
                    stage="metadata_map",
                    attempt=attempt,
                    exc=exc,
                    scope={
                        "chunk_index": chunk.index,
                        "pages": [chunk.page_start, chunk.page_end],
                    },
                )
                if not exc.retryable:
                    raise
                last_error = exc
            if attempt < self.settings.model_max_retries:
                await asyncio.sleep(min(2, 0.25 * (2**attempt)))
        raise ModelError(
            f"Metadata chunk output remained invalid after retries: {last_error}", retryable=True
        ) from last_error

    async def _hierarchical_reduce(
        self,
        document: dict[str, Any],
        mapped: list[ChunkAnalysis],
        candidates: CandidateCatalog,
        progress: ProgressCallback | None,
    ) -> MetadataProposal:
        items: list[dict[str, Any]] = [item.model_dump(mode="json") for item in mapped]
        batch_size = self.settings.metadata_reduce_batch_size
        candidate_size = len(json.dumps(candidates.prompt_dict(), ensure_ascii=False))
        reduction_budget = max(
            4_000,
            (
                self.settings.metadata_context_tokens
                - self.settings.metadata_max_output_tokens
                - 1_000
            )
            * 3
            - candidate_size
            - 5_000,
        )
        level = 0
        while len(items) > 1 and (
            len(items) > batch_size or len(json.dumps(items, ensure_ascii=False)) > reduction_budget
        ):
            groups = _budgeted_groups(items, batch_size, reduction_budget)
            if len(groups) == 1:
                break
            if len(groups) >= len(items):
                raise ModelError(
                    "Metadata findings cannot be compacted inside the configured context; "
                    "increase the metadata context limit"
                )
            reduced: list[dict[str, Any]] = []
            for index, group in enumerate(groups, 1):
                provisional = await self._reduce_group(document, group, candidates, final=False)
                reduced.append(provisional.model_dump(mode="json"))
                if progress:
                    progress(f"metadata_reduce_{level + 1}", index, len(groups))
            items = reduced
            level += 1
        if progress:
            progress("metadata_reduce_final", 0, 1)
        if len(json.dumps(items, ensure_ascii=False)) > reduction_budget:
            raise ModelError(
                "Metadata findings exceed the configured reduction context; increase the metadata context limit"
            )
        final = await self._reduce_group(document, items, candidates, final=True)
        if progress:
            progress("metadata_reduce_final", 1, 1)
        return final

    async def _reduce_group(
        self,
        document: dict[str, Any],
        findings: list[dict[str, Any]],
        candidates: CandidateCatalog,
        *,
        final: bool,
    ) -> MetadataProposal:
        payload = {
            "task": "Produce the final document decision."
            if final
            else "Compact these findings into a provisional decision for a later reduction.",
            "document_id": document.get("id"),
            "controlled_vocabulary": candidates.prompt_dict(),
            "findings": findings,
        }
        user = json.dumps(payload, ensure_ascii=False)
        last_error: Exception | None = None
        for attempt in range(self.settings.model_max_retries + 1):
            try:
                raw = await self.model.structured(
                    name="clerk_metadata_proposal",
                    schema=MetadataProposal.model_json_schema(),
                    system=METADATA_REDUCE_SYSTEM_PROMPT,
                    user=user,
                )
                proposal = MetadataProposal.model_validate(raw)
                if attempt:
                    self._record_diagnostic(
                        {
                            "stage": "metadata_reduce_final" if final else "metadata_reduce",
                            "status": "recovered",
                            "attempt": attempt + 1,
                            "finding_count": len(findings),
                        }
                    )
                return proposal
            except ValidationError as exc:
                last_error = exc
                self._record_validation_failure(
                    stage="metadata_reduce_final" if final else "metadata_reduce",
                    attempt=attempt,
                    exc=exc,
                    raw=raw,
                    scope={"finding_count": len(findings)},
                )
                expected = ", ".join(MetadataProposal.model_fields)
                previous = json.dumps(raw, ensure_ascii=False)[:4_000]
                errors = json.dumps(
                    exc.errors(include_url=False, include_input=False), ensure_ascii=False
                )[:2_000]
                user = (
                    json.dumps(payload, ensure_ascii=False)
                    + "\n\nRepair the previous output instead of repeating it. "
                    + f"Use exactly these top-level keys: {expected}."
                    + f"\nPrevious output: {previous}\nValidation errors: {errors}"
                )
            except ModelError as exc:
                self._record_model_failure(
                    stage="metadata_reduce_final" if final else "metadata_reduce",
                    attempt=attempt,
                    exc=exc,
                    scope={"finding_count": len(findings)},
                )
                if not exc.retryable:
                    raise
                last_error = exc
            if attempt < self.settings.model_max_retries:
                await asyncio.sleep(min(2, 0.25 * (2**attempt)))
        raise ModelError(
            f"Metadata reduction remained invalid after retries: {last_error}", retryable=True
        ) from last_error


@dataclass
class ValidationPlan:
    correspondent: dict[str, Any] | None
    document_type: dict[str, Any] | None
    tags: list[dict[str, Any]]
    title: dict[str, Any] | None
    document_date: dict[str, Any] | None
    custom_fields: list[dict[str, Any]]
    new_custom_fields: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    duplicate_candidates: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "correspondent": self.correspondent,
            "document_type": self.document_type,
            "tags": self.tags,
            "title": self.title,
            "document_date": self.document_date,
            "custom_fields": self.custom_fields,
            "new_custom_fields": self.new_custom_fields,
            "rejected": self.rejected,
            "duplicate_candidates": self.duplicate_candidates,
        }


class MetadataPlanner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(
        self,
        proposal: MetadataProposal,
        catalog: dict[str, list[dict[str, Any]]],
        allowed_ids: dict[str, list[int]] | None = None,
    ) -> ValidationPlan:
        rejected: list[dict[str, Any]] = []
        duplicates: list[dict[str, Any]] = []
        entities = {
            "tag": entities_from_payload(catalog["tags"]),
            "correspondent": entities_from_payload(catalog["correspondents"]),
            "document_type": entities_from_payload(catalog["document_types"]),
        }
        by_id = {kind: {item.id: item for item in values} for kind, values in entities.items()}

        def choice(
            item: MetadataChoice | None,
            kind: str,
            allow_new: bool,
        ) -> dict[str, Any] | None:
            if item is None:
                return None
            data = item.model_dump(mode="json")
            if item.confidence < self.settings.metadata_min_confidence:
                rejected.append(
                    {"kind": kind, "candidate": data, "reason": "below confidence threshold"}
                )
                return None
            if item.existing_id is not None:
                entity = by_id[kind].get(item.existing_id)
                allowed = set(allowed_ids.get(kind, [])) if allowed_ids is not None else None
                if not entity or (allowed is not None and item.existing_id not in allowed):
                    rejected.append(
                        {
                            "kind": kind,
                            "candidate": data,
                            "reason": "unknown or wrong-type existing ID",
                        }
                    )
                    return None
                return {**data, "action": "reuse", "id": entity.id, "name": entity.name}
            assert item.new_name is not None
            if (
                kind == "tag"
                and self.settings.automation_tag
                and " ".join(item.new_name.split()).casefold()
                == self.settings.automation_tag.casefold()
            ):
                rejected.append(
                    {
                        "kind": kind,
                        "candidate": data,
                        "reason": "automation watch tag is workflow-only",
                    }
                )
                return None
            duplicate = find_duplicate(item.new_name, entities[kind], kind)
            explicit_distinction = duplicate and self._explicit_distinction(
                item, duplicate.entity, duplicate.reason
            )
            if duplicate and not explicit_distinction:
                duplicate_record = {
                    "kind": kind,
                    "proposed": item.new_name,
                    "matched_id": duplicate.entity.id,
                    "matched_name": duplicate.entity.name,
                    "score": duplicate.score,
                    "reason": duplicate.reason,
                    "outcome": "normalized_to_existing",
                }
                duplicates.append(duplicate_record)
                return {
                    **data,
                    "action": "reuse",
                    "id": duplicate.entity.id,
                    "name": duplicate.entity.name,
                    "normalized_from": item.new_name,
                }
            if duplicate and explicit_distinction:
                duplicates.append(
                    {
                        "kind": kind,
                        "proposed": item.new_name,
                        "matched_id": duplicate.entity.id,
                        "matched_name": duplicate.entity.name,
                        "score": duplicate.score,
                        "reason": duplicate.reason,
                        "outcome": "created_as_explicit_distinction",
                        "justification": item.reason,
                    }
                )
            if not allow_new:
                rejected.append(
                    {"kind": kind, "candidate": data, "reason": "vocabulary growth disabled"}
                )
                return None
            if len(item.reason.strip()) < 12:
                rejected.append(
                    {
                        "kind": kind,
                        "candidate": data,
                        "reason": "new vocabulary requires a specific justification",
                    }
                )
                return None
            if not name_tokens(item.new_name, kind):
                rejected.append(
                    {"kind": kind, "candidate": data, "reason": "name has no meaningful terms"}
                )
                return None
            selected = {**data, "action": "create", "name": item.new_name}
            if duplicate and explicit_distinction:
                selected["explicit_distinction_from"] = duplicate.entity.id
            return selected

        correspondent = choice(
            proposal.correspondent, "correspondent", self.settings.allow_new_correspondents
        )
        document_type = choice(
            proposal.document_type, "document_type", self.settings.allow_new_document_types
        )
        tags = [
            selected
            for item in proposal.tags
            if (selected := choice(item, "tag", self.settings.allow_new_tags)) is not None
        ]
        tags = self._deduplicate_plan(tags, "tag", duplicates)

        title = self._scalar(proposal.title, "title", rejected)
        document_date = self._date(proposal.document_date, rejected)
        custom_fields = self._custom_fields(proposal.custom_fields, catalog, rejected, allowed_ids)
        new_custom_fields = self._new_custom_fields(
            proposal.new_custom_fields, catalog, rejected, duplicates
        )
        return ValidationPlan(
            correspondent=correspondent,
            document_type=document_type,
            tags=tags,
            title=title,
            document_date=document_date,
            custom_fields=custom_fields,
            new_custom_fields=new_custom_fields,
            rejected=rejected,
            duplicate_candidates=duplicates,
        )

    @staticmethod
    def _explicit_distinction(
        item: MetadataChoice, existing: Entity, duplicate_reason: str
    ) -> bool:
        if (
            duplicate_reason != "existing concept with only a modifier"
            or item.confidence < 0.9
            or len(item.reason.strip()) < 30
        ):
            return False
        reason = item.reason.casefold()
        names_existing = existing.name.casefold() in reason
        contrasts = any(
            phrase in reason
            for phrase in (
                "distinct from",
                "different from",
                "separate from",
                "whereas",
                "not covered by",
                "broader than",
                "narrower than",
            )
        )
        return names_existing and contrasts

    @staticmethod
    def _deduplicate_plan(
        items: list[dict[str, Any]],
        kind: str,
        duplicates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        retained: list[dict[str, Any]] = []
        for item in items:
            duplicate_of: tuple[dict[str, Any], float, str] | None = None
            for existing in retained:
                if item.get("explicit_distinction_from") == existing.get("id") or existing.get(
                    "explicit_distinction_from"
                ) == item.get("id"):
                    continue
                score, reason = duplicate_similarity(item["name"], existing["name"], kind)
                if score >= 0.86:
                    duplicate_of = existing, score, reason
                    break
            if duplicate_of is None:
                retained.append(item)
                continue
            existing, score, reason = duplicate_of
            duplicates.append(
                {
                    "kind": kind,
                    "proposed": item["name"],
                    "matched_id": existing.get("id", existing.get("field_id")),
                    "matched_name": existing["name"],
                    "score": round(score, 4),
                    "reason": f"duplicate within this proposal: {reason}",
                    "outcome": "collapsed_within_proposal",
                }
            )
        return retained

    def _scalar(
        self, item: ScalarCandidate | None, kind: str, rejected: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if item is None:
            return None
        data = item.model_dump(mode="json")
        if item.confidence < self.settings.metadata_min_confidence:
            rejected.append(
                {"kind": kind, "candidate": data, "reason": "below confidence threshold"}
            )
            return None
        return data

    def _date(
        self, item: ScalarCandidate | None, rejected: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        data = self._scalar(item, "document_date", rejected)
        if not data:
            return None
        try:
            parsed = date.fromisoformat(data["value"])
        except ValueError:
            rejected.append(
                {"kind": "document_date", "candidate": data, "reason": "not an ISO date"}
            )
            return None
        if parsed.year < 1900 or parsed > date.today() + timedelta(days=3660):
            rejected.append(
                {"kind": "document_date", "candidate": data, "reason": "implausible date"}
            )
            return None
        data["value"] = parsed.isoformat()
        return data

    def _custom_fields(
        self,
        items: list[CustomFieldCandidate],
        catalog: dict[str, list[dict[str, Any]]],
        rejected: list[dict[str, Any]],
        allowed_ids: dict[str, list[int]] | None,
    ) -> list[dict[str, Any]]:
        definitions = {int(field["id"]): field for field in catalog["custom_fields"]}
        allowed = set(allowed_ids.get("custom_field", [])) if allowed_ids is not None else None
        result: list[dict[str, Any]] = []
        for item in items:
            data = item.model_dump(mode="json")
            definition = definitions.get(item.field_id)
            if not definition or (allowed is not None and item.field_id not in allowed):
                rejected.append(
                    {"kind": "custom_field", "candidate": data, "reason": "unknown field ID"}
                )
                continue
            if item.confidence < self.settings.metadata_min_confidence:
                rejected.append(
                    {
                        "kind": "custom_field",
                        "candidate": data,
                        "reason": "below confidence threshold",
                    }
                )
                continue
            try:
                value = normalize_custom_value(item.value, definition)
            except ValueError as exc:
                rejected.append({"kind": "custom_field", "candidate": data, "reason": str(exc)})
                continue
            result.append(
                {
                    **data,
                    "field_id": item.field_id,
                    "field_name": definition["name"],
                    "value": value,
                }
            )
        return list({item["field_id"]: item for item in result}.values())

    def _new_custom_fields(
        self,
        items: list[NewCustomFieldCandidate],
        catalog: dict[str, list[dict[str, Any]]],
        rejected: list[dict[str, Any]],
        duplicates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.settings.allow_new_custom_fields:
            rejected.extend(
                {
                    "kind": "new_custom_field",
                    "candidate": item.model_dump(mode="json"),
                    "reason": "new definitions disabled",
                }
                for item in items
            )
            return []
        existing = entities_from_payload(catalog["custom_fields"])
        result: list[dict[str, Any]] = []
        for item in items:
            data = item.model_dump(mode="json")
            if item.confidence < self.settings.metadata_min_confidence:
                rejected.append(
                    {
                        "kind": "new_custom_field",
                        "candidate": data,
                        "reason": "below confidence threshold",
                    }
                )
                continue
            if len(item.reason.strip()) < 12:
                rejected.append(
                    {
                        "kind": "new_custom_field",
                        "candidate": data,
                        "reason": "new field definitions require a specific justification",
                    }
                )
                continue
            duplicate = find_duplicate(item.name, existing, "custom_field")
            if duplicate:
                duplicates.append(
                    {
                        "kind": "custom_field",
                        "proposed": item.name,
                        "matched_id": duplicate.entity.id,
                        "matched_name": duplicate.entity.name,
                        "score": duplicate.score,
                        "reason": duplicate.reason,
                        "outcome": "reused_existing_field",
                    }
                )
                definition = duplicate.entity.raw or {}
                try:
                    value = normalize_custom_value(item.value, definition)
                except ValueError as exc:
                    rejected.append(
                        {"kind": "new_custom_field", "candidate": data, "reason": str(exc)}
                    )
                    continue
                result.append(
                    {**data, "action": "reuse", "field_id": duplicate.entity.id, "value": value}
                )
            else:
                try:
                    value = normalize_custom_value(
                        item.value, {"data_type": item.data_type, "extra_data": {}}
                    )
                except ValueError as exc:
                    rejected.append(
                        {"kind": "new_custom_field", "candidate": data, "reason": str(exc)}
                    )
                    continue
                result.append({**data, "action": "create", "value": value})
        return self._deduplicate_plan(result, "custom_field", duplicates)


async def apply_metadata_plan(
    *,
    plan: ValidationPlan,
    document: dict[str, Any],
    paperless: PaperlessClient,
    settings: Settings,
    consume_tag: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve creations, construct one conservative PATCH, and apply it."""

    patch: dict[str, Any] = {}
    applied: dict[str, Any] = {
        "reused": [],
        "created": [],
        "removed": [],
        "assignments": [],
        "withheld": [],
        "patch": patch,
    }

    def rationale(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in ("confidence", "reason", "evidence", "source_pages", "normalized_from")
            if item.get(key) not in (None, "", [])
        }

    async def resolve_entity(item: dict[str, Any], resource: str) -> int:
        if item["action"] == "reuse":
            applied["reused"].append(
                {
                    "resource": resource,
                    "id": item["id"],
                    "name": item["name"],
                    **rationale(item),
                }
            )
            return int(item["id"])
        created = await paperless.ensure_entity(resource, item["name"])
        applied["created"].append(
            {
                "resource": resource,
                "id": created["id"],
                "name": created["name"],
                **rationale(item),
            }
        )
        return int(created["id"])

    original_tag_ids = {int(tag_id) for tag_id in document.get("tags", [])}
    tag_ids = set(original_tag_ids)
    if consume_tag and int(consume_tag["id"]) in tag_ids:
        tag_ids.remove(int(consume_tag["id"]))
        applied["removed"].append(
            {
                "resource": "tags",
                "id": int(consume_tag["id"]),
                "name": str(consume_tag.get("name") or settings.automation_tag),
                "reason": "automation watch tag consumed after successful metadata processing",
            }
        )
    for tag in plan.tags:
        tag_ids.add(await resolve_entity(tag, "tags"))
    if tag_ids != original_tag_ids:
        patch["tags"] = sorted(tag_ids)

    overwrite = settings.metadata_apply_mode == "overwrite"
    if plan.correspondent:
        if not document.get("correspondent") or overwrite:
            patch["correspondent"] = await resolve_entity(plan.correspondent, "correspondents")
        else:
            applied["withheld"].append(
                {"field": "correspondent", "reason": "existing value preserved"}
            )
    if plan.document_type:
        if not document.get("document_type") or overwrite:
            patch["document_type"] = await resolve_entity(plan.document_type, "document_types")
        else:
            applied["withheld"].append(
                {"field": "document_type", "reason": "existing value preserved"}
            )

    if plan.title:
        if overwrite or _generic_title(document):
            patch["title"] = plan.title["value"][:128]
            applied["assignments"].append(
                {"field": "title", "value": patch["title"], **rationale(plan.title)}
            )
        else:
            applied["withheld"].append(
                {"field": "title", "reason": "non-generic existing title preserved"}
            )
    if plan.document_date:
        current_date = str(document.get("created") or document.get("created_date") or "")[:10]
        added_date = str(document.get("added") or "")[:10]
        if overwrite or not current_date or current_date == added_date:
            patch["created"] = plan.document_date["value"]
            applied["assignments"].append(
                {
                    "field": "document_date",
                    "value": patch["created"],
                    **rationale(plan.document_date),
                }
            )
        else:
            applied["withheld"].append(
                {"field": "created", "reason": "existing intrinsic date preserved"}
            )

    custom_values = {
        int(item["field"]): {"field": int(item["field"]), "value": item.get("value")}
        for item in document.get("custom_fields", [])
        if item.get("field") is not None
    }
    custom_changed = False
    for item in plan.custom_fields:
        field_id = int(item["field_id"])
        if (
            overwrite
            or field_id not in custom_values
            or custom_values[field_id]["value"] in (None, "")
        ):
            custom_values[field_id] = {"field": field_id, "value": item["value"]}
            custom_changed = True
            applied["assignments"].append(
                {
                    "field": f"custom_field:{field_id}",
                    "field_name": item["field_name"],
                    "value": item["value"],
                    **rationale(item),
                }
            )
        else:
            applied["withheld"].append(
                {"field": f"custom_field:{field_id}", "reason": "existing value preserved"}
            )
    for item in plan.new_custom_fields:
        if item["action"] == "reuse":
            field_id = int(item["field_id"])
        else:
            definition = await paperless.create_custom_field(item["name"], item["data_type"])
            field_id = int(definition["id"])
            applied["created"].append(
                {
                    "resource": "custom_fields",
                    "id": field_id,
                    "name": item["name"],
                    **rationale(item),
                }
            )
        if overwrite or field_id not in custom_values:
            custom_values[field_id] = {"field": field_id, "value": item["value"]}
            custom_changed = True
            applied["assignments"].append(
                {
                    "field": f"custom_field:{field_id}",
                    "field_name": item["name"],
                    "value": item["value"],
                    **rationale(item),
                }
            )
    if custom_changed:
        patch["custom_fields"] = list(custom_values.values())

    updated = await paperless.update_document(int(document["id"]), patch) if patch else document
    return applied, updated


def _generic_title(document: dict[str, Any]) -> bool:
    title = str(document.get("title") or "").strip()
    if not title or title.casefold() in {"untitled", "document", "scan", "scanned document"}:
        return True
    original = Path(str(document.get("original_file_name") or "")).stem

    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    return bool(original and normalize(title) == normalize(original))


def _compact_candidate(candidate: Any) -> dict[str, Any] | None:
    if candidate is None:
        return None
    data = candidate.model_dump(mode="json")
    if data.get("reason"):
        data["reason"] = str(data["reason"])[:240]
    if data.get("evidence"):
        data["evidence"] = str(data["evidence"])[:240]
    if data.get("source_pages"):
        data["source_pages"] = data["source_pages"][:12]
    return data


def _representative_text(pages: list[tuple[int, str]], maximum_characters: int) -> str:
    """Sample long documents across their span without building a second giant prompt."""

    available = [(number, text.strip()) for number, text in sorted(pages) if text.strip()]
    if not available or maximum_characters <= 0:
        return ""
    complete = "\n\n".join(f"[Page {number}]\n{text}" for number, text in available)
    if len(complete) <= maximum_characters:
        return complete

    maximum_samples = min(8, len(available))
    if maximum_samples == 1:
        selected_indices = [0]
    else:
        selected_indices = sorted(
            {
                round(index * (len(available) - 1) / (maximum_samples - 1))
                for index in range(maximum_samples)
            }
        )
    per_page = max(128, maximum_characters // len(selected_indices) - 24)
    samples: list[str] = []
    for index in selected_indices:
        number, text = available[index]
        if len(text) > per_page:
            tail_size = max(32, per_page // 4)
            text = f"{text[: per_page - tail_size - 5]} [...] {text[-tail_size:]}"
        samples.append(f"[Page {number}]\n{text}")
    return "\n\n".join(samples)[:maximum_characters]


def normalize_custom_value(value: Any, definition: dict[str, Any]) -> Any:
    data_type = definition.get("data_type")
    if value is None:
        return None
    if data_type == "string":
        text = " ".join(str(value).split())
        if len(text) > 128:
            raise ValueError("string value exceeds 128 characters")
        return text
    if data_type == "longtext":
        return str(value).strip()
    if data_type == "boolean":
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().casefold()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
        raise ValueError("invalid boolean value")
    if data_type == "integer":
        try:
            result = int(str(value).replace(",", ""))
        except ValueError as exc:
            raise ValueError("invalid integer value") from exc
        if not -(2**31) <= result < 2**31:
            raise ValueError("integer value is outside Paperless's supported range")
        return result
    if data_type == "float":
        try:
            result = float(str(value).replace(",", ""))
        except ValueError as exc:
            raise ValueError("invalid floating-point value") from exc
        if not math.isfinite(result):
            raise ValueError("floating-point value must be finite")
        return result
    if data_type == "date":
        try:
            return date.fromisoformat(str(value)[:10]).isoformat()
        except ValueError as exc:
            raise ValueError("invalid ISO date value") from exc
    if data_type == "url":
        text = str(value).strip()
        if not text.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return text
    if data_type == "monetary":
        text = str(value).strip().replace(",", "")
        match = re.fullmatch(r"([A-Za-z]{3})?\s*([-+]?\d+(?:\.\d{1,2})?)", text)
        if not match:
            raise ValueError("invalid monetary value")
        currency = (
            match.group(1) or (definition.get("extra_data") or {}).get("default_currency") or ""
        ).upper()
        number = float(match.group(2))
        return f"{currency}{number:.2f}" if currency else f"{number:.2f}"
    if data_type == "select":
        options = (definition.get("extra_data") or {}).get("select_options", [])
        raw = str(value).strip()
        match = next(
            (
                option
                for option in options
                if str(option.get("id")) == raw
                or str(option.get("label", "")).casefold() == raw.casefold()
            ),
            None,
        )
        if not match:
            raise ValueError("value is not an existing select option")
        return match["id"]
    if data_type == "documentlink":
        # Clerk does not put arbitrary documents into a model prompt, so it has
        # no bounded, validated ID set from which a document link may be chosen.
        raise ValueError("document-link values require manual assignment")
    raise ValueError(f"unsupported custom-field type: {data_type}")


def _budgeted_groups(
    items: list[dict[str, Any]], maximum_items: int, maximum_characters: int
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 2
    for item in items:
        item_size = len(json.dumps(item, ensure_ascii=False)) + 1
        if current and (
            len(current) >= maximum_items or current_size + item_size > maximum_characters
        ):
            groups.append(current)
            current, current_size = [], 2
        current.append(item)
        current_size += item_size
    if current:
        groups.append(current)
    return groups
