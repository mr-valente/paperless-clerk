import asyncio
import json

import pytest

from paperless_clerk.config import Settings
from paperless_clerk.metadata import MetadataAnalyzer, MetadataPlanner, _representative_text
from paperless_clerk.schemas import ChunkAnalysis, MetadataChoice, MetadataProposal


def gate_create_verdicts(user: str) -> dict:
    """Default vocabulary-gate answer for fakes: approve every proposal unchanged."""
    payload = json.loads(user)
    return {
        "verdicts": [
            {
                "kind": item["kind"],
                "proposed_name": item["name"],
                "decision": "create",
                "existing_id": None,
                "name": None,
                "reason": "Distinct reusable concept",
            }
            for item in payload.get("proposals", [])
        ]
    }


class FakeStructuredModel:
    def __init__(self):
        self.map_calls = 0
        self.reduce_calls = 0
        self.active_maps = 0
        self.max_active_maps = 0
        self.tag_review_calls = 0

    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            return gate_create_verdicts(user)
        if name == "clerk_chunk_analysis":
            self.map_calls += 1
            self.active_maps += 1
            self.max_active_maps = max(self.max_active_maps, self.active_maps)
            await asyncio.sleep(0.001)
            self.active_maps -= 1
            return {}
        if name in {"clerk_tag_review", "clerk_tag_abstention_audit"}:
            self.tag_review_calls += 1
            return {
                "tags": [],
                "assessment": "No supported reusable tag was found in the supplied vocabulary.",
            }
        self.reduce_calls += 1
        return {"summary": "Conservative aggregate"}


class RepairingStructuredModel(FakeStructuredModel):
    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            return gate_create_verdicts(user)
        if name == "clerk_chunk_analysis" and self.map_calls == 0:
            self.map_calls += 1
            return {"unexpected": "field"}
        return await super().structured(name=name, schema=schema, system=system, user=user)


class FlatThenSchemaCorrectModel(FakeStructuredModel):
    def __init__(self):
        super().__init__()
        self.repair_prompt = ""

    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            return gate_create_verdicts(user)
        if name == "clerk_chunk_analysis":
            self.map_calls += 1
            if self.map_calls == 1:
                return {
                    "title": "Faculty Employment Agreement — School of the Holy Child",
                    "correspondent_id": None,
                    "document_type_id": None,
                    "tag_ids": [55],
                    "date": "2025-03-09",
                    "custom_fields": [],
                }
            self.repair_prompt = user
            return {
                "correspondent_candidates": [],
                "document_type_candidates": [],
                "tag_candidates": [
                    {
                        "existing_id": 55,
                        "confidence": 0.91,
                        "reason": "Employment is a useful existing retrieval category",
                        "evidence": "Faculty Employment Agreement",
                        "source_pages": [1],
                    }
                ],
                "title_candidates": [
                    {
                        "value": "Faculty Employment Agreement — School of the Holy Child",
                        "confidence": 0.96,
                        "reason": "The document heading identifies the agreement",
                        "evidence": "Faculty Employment Agreement",
                        "source_pages": [1],
                    }
                ],
                "date_candidates": [
                    {
                        "value": "2025-03-09",
                        "confidence": 0.9,
                        "reason": "The agreement states this intrinsic date",
                        "evidence": "March 9, 2025",
                        "source_pages": [1],
                    }
                ],
                "custom_field_candidates": [],
                "new_custom_field_candidates": [],
            }

        self.reduce_calls += 1
        finding = json.loads(user)["findings"][0]
        return {
            "tags": finding["tag_candidates"],
            "title": finding["title_candidates"][0],
            "document_date": finding["date_candidates"][0],
            "summary": "Repaired and reduced chunk output",
        }


class FocusedTagReviewModel(FakeStructuredModel):
    def __init__(self):
        super().__init__()
        self.review_payload: dict = {}

    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            return gate_create_verdicts(user)
        if name == "clerk_chunk_analysis":
            self.map_calls += 1
            return {
                "document_type_candidates": [
                    {
                        "new_name": "Employment Agreement",
                        "confidence": 0.95,
                        "reason": "The document is explicitly an employment agreement",
                        "evidence": "Faculty Employment Agreement",
                        "source_pages": [1],
                    }
                ],
                "title_candidates": [
                    {
                        "value": "Faculty Employment Agreement",
                        "confidence": 0.96,
                        "reason": "The heading states the document title",
                        "evidence": "Faculty Employment Agreement",
                        "source_pages": [1],
                    }
                ],
            }
        if name == "clerk_tag_review":
            self.tag_review_calls += 1
            self.review_payload = json.loads(user)
            return {
                "tags": [
                    {
                        "existing_id": 55,
                        "confidence": 0.93,
                        "reason": "Employment is the canonical cross-cutting library category",
                        "evidence": "Faculty Employment Agreement",
                        "source_pages": [1],
                    }
                ],
                "assessment": "A focused review found the existing Employment tag applicable.",
            }
        self.reduce_calls += 1
        return {
            "document_type": {
                "new_name": "Employment Agreement",
                "confidence": 0.95,
                "reason": "The document is explicitly an employment agreement",
                "evidence": "Faculty Employment Agreement",
                "source_pages": [1],
            },
            "title": {
                "value": "Faculty Employment Agreement",
                "confidence": 0.96,
                "reason": "The heading states the document title",
                "evidence": "Faculty Employment Agreement",
                "source_pages": [1],
            },
            "summary": "Classified the agreement, but the general pass omitted tags.",
        }


class VerboseFocusedTagReviewModel(FocusedTagReviewModel):
    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            return gate_create_verdicts(user)
        result = await super().structured(name=name, schema=schema, system=system, user=user)
        if name == "clerk_tag_review":
            result["assessment"] = (
                "The document is a substantive veterinary invoice and the selected canonical "
                "tag improves retrieval without introducing an unsupported category. "
            ) * 12
        return result


class RepairingFocusedTagReviewModel(FocusedTagReviewModel):
    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            return gate_create_verdicts(user)
        if name == "clerk_tag_review":
            self.tag_review_calls += 1
            if self.tag_review_calls == 1:
                return {
                    "tags": [],
                    "assessment": "The first response included an unsupported top-level field.",
                    "unsupported": True,
                }
            return {
                "tags": [
                    {
                        "existing_id": 55,
                        "confidence": 0.93,
                        "reason": "Employment is the canonical cross-cutting library category",
                        "evidence": "Faculty Employment Agreement",
                        "source_pages": [1],
                    }
                ],
                "assessment": "The repaired response selected the existing Employment tag.",
            }
        return await super().structured(name=name, schema=schema, system=system, user=user)


class VeterinaryAbstentionAuditModel(FakeStructuredModel):
    def __init__(self):
        super().__init__()
        self.audit_payload: dict = {}
        self.audit_system = ""

    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            return gate_create_verdicts(user)
        if name == "clerk_chunk_analysis":
            self.map_calls += 1
            return {
                "document_type_candidates": [
                    {
                        "existing_id": 7,
                        "confidence": 0.99,
                        "reason": "The document is explicitly an invoice for veterinary services",
                        "evidence": "Invoice Date",
                        "source_pages": [1],
                    }
                ]
            }
        if name == "clerk_tag_review":
            self.tag_review_calls += 1
            return {
                "tags": [],
                "assessment": (
                    "There is no existing Veterinary or Medical tag. Veterinary appears useful, "
                    "but this is the first veterinary invoice and Invoice is already the type."
                ),
            }
        if name == "clerk_tag_abstention_audit":
            self.tag_review_calls += 1
            self.audit_payload = json.loads(user)
            self.audit_system = system
            return {
                "tags": [
                    {
                        "new_name": "Veterinary",
                        "confidence": 0.96,
                        "reason": (
                            "Veterinary is a broad reusable subject spanning invoices, "
                            "medical records, prescriptions, and lab results"
                        ),
                        "evidence": "veterinary services",
                        "source_pages": [1],
                    }
                ],
                "assessment": (
                    "Created Veterinary as the reusable subject; Invoice remains the document type."
                ),
            }
        self.reduce_calls += 1
        return {
            "document_type": {
                "existing_id": 7,
                "confidence": 0.99,
                "reason": "The document is explicitly an invoice for veterinary services",
                "evidence": "Invoice Date",
                "source_pages": [1],
            },
            "summary": "Veterinary invoice from an animal hospital.",
        }


@pytest.mark.asyncio
async def test_large_metadata_analysis_uses_bounded_hierarchical_map_reduce() -> None:
    settings = Settings(
        metadata_chunk_chars=2000,
        metadata_reduce_batch_size=4,
        metadata_concurrency=3,
    )
    model = FakeStructuredModel()
    analyzer = MetadataAnalyzer(settings, model)  # type: ignore[arg-type]
    pages = [
        (page, f"Unique page {page}. " + ("statement content " * 180)) for page in range(1, 101)
    ]
    catalog = {"tags": [], "correspondents": [], "document_types": [], "custom_fields": []}

    proposal, trace = await analyzer.analyze(
        document={"id": 77, "tags": [], "title": "Statement"}, pages=pages, catalog=catalog
    )

    assert model.map_calls == len(trace["chunks"])
    assert model.map_calls > 100
    assert model.reduce_calls > 1
    assert model.max_active_maps <= 3
    assert proposal.summary == "Conservative aggregate"


@pytest.mark.asyncio
async def test_invalid_structured_map_output_is_retried_with_schema_feedback() -> None:
    model = RepairingStructuredModel()
    analyzer = MetadataAnalyzer(
        Settings(model_max_retries=1, metadata_chunk_chars=2000),
        model,  # type: ignore[arg-type]
    )
    catalog = {"tags": [], "correspondents": [], "document_types": [], "custom_fields": []}

    proposal, _ = await analyzer.analyze(
        document={"id": 88, "tags": []},
        pages=[(1, "A sufficiently long invoice with an issue date and sender information.")],
        catalog=catalog,
    )

    assert model.map_calls == 2
    assert proposal.summary == "Conservative aggregate"


@pytest.mark.asyncio
async def test_flat_document_shape_is_repaired_into_chunk_candidates() -> None:
    model = FlatThenSchemaCorrectModel()
    analyzer = MetadataAnalyzer(
        Settings(model_max_retries=1, metadata_chunk_chars=2000),
        model,  # type: ignore[arg-type]
    )
    catalog = {
        "tags": [{"id": 55, "name": "Employment", "document_count": 3}],
        "correspondents": [],
        "document_types": [],
        "custom_fields": [],
    }

    proposal, _ = await analyzer.analyze(
        document={"id": 89, "tags": []},
        pages=[(1, "Faculty Employment Agreement dated March 9, 2025.")],
        catalog=catalog,
    )

    assert model.map_calls == 2
    assert "Repair the previous output" in model.repair_prompt
    assert "title_candidates" in model.repair_prompt
    assert "Faculty Employment Agreement" in model.repair_prompt
    assert proposal.tags[0].existing_id == 55
    assert proposal.title is not None
    assert proposal.title.value == "Faculty Employment Agreement — School of the Holy Child"


@pytest.mark.asyncio
async def test_empty_general_tag_result_gets_a_focused_second_pass() -> None:
    model = FocusedTagReviewModel()
    analyzer = MetadataAnalyzer(Settings(metadata_chunk_chars=2000), model)  # type: ignore[arg-type]
    catalog = {
        "tags": [{"id": 55, "name": "Employment", "document_count": 3}],
        "correspondents": [],
        "document_types": [],
        "custom_fields": [],
    }

    proposal, trace = await analyzer.analyze(
        document={"id": 90, "tags": []},
        pages=[(1, "Faculty Employment Agreement dated March 9, 2025.")],
        catalog=catalog,
    )

    assert model.tag_review_calls == 1
    assert proposal.tags[0].existing_id == 55
    assert trace["tag_review"] == {
        "performed": True,
        "status": "selected",
        "assessment": "A focused review found the existing Employment tag applicable.",
        "selected_count": 1,
        "new_tags_allowed": True,
        "abstention_audit_performed": False,
    }
    assert model.review_payload["controlled_vocabulary"]["tags"][0]["id"] == 55
    assert "Faculty Employment Agreement" in model.review_payload["representative_document_text"]


@pytest.mark.asyncio
async def test_verbose_tag_assessment_does_not_discard_valid_tag_choices() -> None:
    model = VerboseFocusedTagReviewModel()
    analyzer = MetadataAnalyzer(
        Settings(metadata_chunk_chars=2000, model_max_retries=0),
        model,  # type: ignore[arg-type]
    )
    catalog = {
        "tags": [{"id": 55, "name": "Veterinary", "document_count": 3}],
        "correspondents": [],
        "document_types": [],
        "custom_fields": [],
    }

    proposal, trace = await analyzer.analyze(
        document={"id": 91, "tags": []},
        pages=[(1, "Veterinary invoice for an examination and laboratory services.")],
        catalog=catalog,
    )

    assert model.tag_review_calls == 1
    assert proposal.tags[0].existing_id == 55
    assert len(trace["tag_review"]["assessment"]) == 700
    assert trace["tag_review"]["assessment"].endswith("...")
    diagnostic = next(
        item for item in trace["model_diagnostics"] if item["stage"] == "focused_tag_review"
    )
    assert diagnostic["status"] == "normalized"
    assert diagnostic["returned_tag_count"] == 1
    assert diagnostic["assessment_original_chars"] > diagnostic["assessment_stored_chars"]


@pytest.mark.asyncio
async def test_tag_review_validation_repair_is_retained_in_diagnostics() -> None:
    model = RepairingFocusedTagReviewModel()
    analyzer = MetadataAnalyzer(
        Settings(metadata_chunk_chars=2000, model_max_retries=1),
        model,  # type: ignore[arg-type]
    )
    catalog = {
        "tags": [{"id": 55, "name": "Employment", "document_count": 3}],
        "correspondents": [],
        "document_types": [],
        "custom_fields": [],
    }

    proposal, trace = await analyzer.analyze(
        document={"id": 92, "tags": []},
        pages=[(1, "Faculty Employment Agreement dated March 9, 2025.")],
        catalog=catalog,
    )

    assert proposal.tags[0].existing_id == 55
    failure = next(
        item
        for item in trace["model_diagnostics"]
        if item["stage"] == "focused_tag_review" and item["status"] == "validation_error"
    )
    completion = next(
        item
        for item in reversed(trace["model_diagnostics"])
        if item["stage"] == "focused_tag_review"
    )
    assert failure["errors"][0]["path"] == "unsupported"
    assert '"unsupported":true' in failure["output_preview"]
    assert completion["status"] == "completed"
    assert completion["recovered_after_validation_error"] is True


@pytest.mark.asyncio
async def test_first_veterinary_invoice_abstention_is_audited_into_broad_new_tag() -> None:
    model = VeterinaryAbstentionAuditModel()
    settings = Settings(metadata_chunk_chars=2000, allow_new_tags=True)
    analyzer = MetadataAnalyzer(settings, model)  # type: ignore[arg-type]
    catalog = {
        "tags": [{"id": 50, "name": "Employment", "document_count": 5}],
        "correspondents": [],
        "document_types": [{"id": 7, "name": "Invoice", "document_count": 12}],
        "custom_fields": [],
    }

    proposal, trace = await analyzer.analyze(
        document={"id": 93, "tags": [], "title": "Animal Hospital Invoice"},
        pages=[
            (
                1,
                "Morristown Animal Hospital veterinary services invoice for examination and labs.",
            )
        ],
        catalog=catalog,
    )
    plan = MetadataPlanner(settings).validate(proposal, catalog, allowed_ids=trace["candidate_ids"])

    assert model.tag_review_calls == 2
    assert proposal.tags[0].new_name == "Veterinary"
    assert plan.tags[0]["action"] == "create"
    assert plan.tags[0]["name"] == "Veterinary"
    assert trace["tag_review"]["status"] == "selected"
    assert trace["tag_review"]["abstention_audit_performed"] is True
    assert trace["tag_review"]["abstention_audit_status"] == "overturned"
    assert "first veterinary invoice" in trace["tag_review"]["initial_assessment"]
    assert model.audit_payload["new_tags_allowed"] is True
    assert model.audit_payload["prior_focused_review"]["tags"] == []
    assert "Veterinary" in model.audit_system
    assert trace["candidate_vocabulary"]["tags"] == [{"id": 50, "name": "Employment", "usage": 5}]
    # The new Veterinary tag then passes through the vocabulary gate.
    assert [item["stage"] for item in trace["model_diagnostics"][-3:]] == [
        "focused_tag_review",
        "tag_abstention_audit",
        "vocabulary_review",
    ]
    assert trace["vocabulary_review"]["verdicts"] == [
        {
            "kind": "tag",
            "proposed": "Veterinary",
            "reason": "Distinct reusable concept",
            "outcome": "created",
        }
    ]


def test_focused_tag_review_samples_large_documents_within_budget() -> None:
    pages = [(page, f"Page {page} content " * 500) for page in range(1, 101)]

    excerpt = _representative_text(pages, 2_000)

    assert len(excerpt) <= 2_000
    assert "[Page 1]" in excerpt
    assert "[Page 100]" in excerpt
    assert excerpt.count("[Page ") <= 8


class ValidatorErrorThenValidModel(FakeStructuredModel):
    """First map output trips a model_validator ValueError, not a schema mismatch."""

    def __init__(self):
        super().__init__()
        self.repair_prompt = ""

    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            return gate_create_verdicts(user)
        if name == "clerk_chunk_analysis":
            self.map_calls += 1
            if self.map_calls == 1:
                return {
                    "tag_candidates": [
                        {
                            "existing_id": 55,
                            "new_name": "Also a new name",
                            "confidence": 0.9,
                            "reason": "Both targets supplied",
                        }
                    ]
                }
            self.repair_prompt = user
            return {}
        return await super().structured(name=name, schema=schema, system=system, user=user)


@pytest.mark.asyncio
async def test_validator_value_error_is_rendered_into_repair_prompt() -> None:
    # pydantic stores the raw ValueError raised by a validator in the error ``ctx``;
    # the repair prompt must never hand that object to json.dumps.
    model = ValidatorErrorThenValidModel()
    analyzer = MetadataAnalyzer(
        Settings(model_max_retries=1, metadata_chunk_chars=2000),
        model,  # type: ignore[arg-type]
    )
    catalog = {"tags": [], "correspondents": [], "document_types": [], "custom_fields": []}

    proposal, _ = await analyzer.analyze(
        document={"id": 89, "tags": []},
        pages=[(1, "A sufficiently long invoice with an issue date and sender information.")],
        catalog=catalog,
    )

    assert model.map_calls == 2
    assert "provide exactly one of existing_id or new_name" in model.repair_prompt
    assert proposal.summary == "Conservative aggregate"


def test_prompt_schema_requires_nullable_target_fields() -> None:
    # Grammar-constrained servers let the model skip optional properties, so a
    # nullable field must be required to force an explicit null.
    from paperless_clerk.schemas import model_prompt_schema

    chunk_schema = model_prompt_schema(ChunkAnalysis)
    choice = chunk_schema["$defs"]["MetadataChoice"]
    assert {"existing_id", "new_name", "confidence", "reason"} <= set(choice["required"])
    assert "evidence" not in choice["required"]
    # ChunkAnalysis itself only has list fields with defaults, so nothing is forced there.
    assert "required" not in chunk_schema

    proposal_schema = model_prompt_schema(MetadataProposal)
    assert {"correspondent", "document_type", "title", "document_date"} <= set(
        proposal_schema["required"]
    )
    # The pydantic models themselves are untouched: explicit nulls still validate.
    MetadataChoice(existing_id=4, new_name=None, confidence=0.9, reason="r")
    assert "existing_id" not in MetadataChoice.model_json_schema().get("required", [])


class SynonymGateModel(FakeStructuredModel):
    """The general pass invents Optometry and a correspondent alias; the gate sees the library."""

    def __init__(self):
        super().__init__()
        self.gate_payload: dict = {}
        self.gate_system = ""

    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            self.gate_payload = json.loads(user)
            self.gate_system = system
            return {
                "verdicts": [
                    {
                        "kind": "tag",
                        "proposed_name": "Optometry",
                        "decision": "reuse",
                        "existing_id": 62,
                        "name": None,
                        "reason": "Optical already collects eye-care documents",
                    },
                    {
                        "kind": "tag",
                        "proposed_name": "Postal Service",
                        "decision": "drop",
                        "existing_id": None,
                        "name": None,
                        "reason": "Restates the correspondent",
                    },
                    {
                        "kind": "tag",
                        "proposed_name": "medical records",
                        "decision": "create",
                        "existing_id": None,
                        "name": "Health",
                        "reason": "Broad durable subject",
                    },
                    {
                        "kind": "correspondent",
                        "proposed_name": "USPS",
                        "decision": "reuse",
                        "existing_id": 13,
                        "name": None,
                        "reason": "Acronym of the existing correspondent",
                    },
                ]
            }
        if name == "clerk_chunk_analysis":
            self.map_calls += 1
            return {}
        self.reduce_calls += 1
        return {
            "correspondent": {
                "new_name": "USPS",
                "confidence": 0.9,
                "reason": "The letter is from the postal service",
                "evidence": "USPS",
                "source_pages": [1],
            },
            "tags": [
                {
                    "existing_id": 62,
                    "confidence": 0.9,
                    "reason": "Eye care subject already in use",
                    "source_pages": [1],
                },
                {
                    "new_name": "Optometry",
                    "confidence": 0.9,
                    "reason": "Issued by an eye doctor",
                    "source_pages": [1],
                },
                {
                    "new_name": "Postal Service",
                    "confidence": 0.9,
                    "reason": "Mailed by the postal service",
                    "source_pages": [1],
                },
                {
                    "new_name": "medical records",
                    "confidence": 0.9,
                    "reason": "Medical record for the patient",
                    "source_pages": [1],
                },
            ],
            "summary": "Eye prescription mailed by USPS",
        }


@pytest.mark.asyncio
async def test_vocabulary_gate_reuses_drops_and_canonicalizes_new_names() -> None:
    model = SynonymGateModel()
    settings = Settings(metadata_chunk_chars=2000, metadata_candidate_limit=10)
    analyzer = MetadataAnalyzer(settings, model)  # type: ignore[arg-type]
    # More tags than the candidate limit, so Optical can fall outside the ranked subset.
    filler = [
        {"id": 100 + index, "name": f"Subject {index}", "document_count": 50} for index in range(12)
    ]
    catalog = {
        "tags": [*filler, {"id": 62, "name": "Optical", "document_count": 1}],
        "correspondents": [{"id": 13, "name": "United States Postal Service", "document_count": 2}],
        "document_types": [],
        "custom_fields": [],
    }

    proposal, trace = await analyzer.analyze(
        document={"id": 19, "tags": [], "title": "Scan"},
        pages=[(1, "Contact lens prescription mailed by the United States Postal Service.")],
        catalog=catalog,
    )

    assert proposal.correspondent is not None
    assert (proposal.correspondent.existing_id, proposal.correspondent.new_name) == (13, None)
    assert [(tag.existing_id, tag.new_name) for tag in proposal.tags] == [
        (62, None),
        (None, "Health"),
    ]
    outcomes = {
        record["proposed"]: record["outcome"] for record in trace["vocabulary_review"]["verdicts"]
    }
    assert outcomes == {
        "USPS": "reused",
        "Optometry": "reused",
        "Postal Service": "dropped",
        "medical records": "renamed",
    }
    # The gate saw the whole library, not the ranked subset, and the planner may now reuse 62.
    assert len(model.gate_payload["existing_vocabulary"]["tags"]) == 13
    assert 62 in trace["candidate_ids"]["tag"]
    assert 13 in trace["candidate_ids"]["correspondent"]
    assert "Tag guidance:" in model.gate_system
    assert "Health" in model.gate_system
    plan = MetadataPlanner(settings).validate(proposal, catalog, allowed_ids=trace["candidate_ids"])
    assert [(item["action"], item["name"]) for item in plan.tags] == [
        ("reuse", "Optical"),
        ("create", "Health"),
    ]
    assert plan.correspondent == {**plan.correspondent, "action": "reuse", "id": 13}


class BrokenGateModel(SynonymGateModel):
    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        if name == "clerk_vocabulary_review":
            return {"unexpected": True}
        return await super().structured(name=name, schema=schema, system=system, user=user)


@pytest.mark.asyncio
async def test_vocabulary_gate_failure_keeps_the_proposal_for_the_lexical_planner() -> None:
    model = BrokenGateModel()
    analyzer = MetadataAnalyzer(Settings(metadata_chunk_chars=2000, model_max_retries=1), model)  # type: ignore[arg-type]
    catalog = {
        "tags": [{"id": 62, "name": "Optical", "document_count": 1}],
        "correspondents": [{"id": 13, "name": "United States Postal Service", "document_count": 2}],
        "document_types": [],
        "custom_fields": [],
    }

    proposal, trace = await analyzer.analyze(
        document={"id": 19, "tags": [], "title": "Scan"},
        pages=[(1, "Contact lens prescription mailed by the United States Postal Service.")],
        catalog=catalog,
    )

    assert trace["vocabulary_review"]["status"] == "failed"
    assert [tag.new_name for tag in proposal.tags] == [
        None,
        "Optometry",
        "Postal Service",
        "medical records",
    ]
    assert proposal.correspondent is not None and proposal.correspondent.new_name == "USPS"


@pytest.mark.asyncio
async def test_vocabulary_gate_is_skipped_without_new_names() -> None:
    model = FakeStructuredModel()
    analyzer = MetadataAnalyzer(Settings(metadata_chunk_chars=2000), model)  # type: ignore[arg-type]
    catalog = {"tags": [], "correspondents": [], "document_types": [], "custom_fields": []}

    _, trace = await analyzer.analyze(
        document={"id": 1, "tags": [5]},
        pages=[(1, "A sufficiently long document with nothing new to propose.")],
        catalog=catalog,
    )

    assert trace["vocabulary_review"] == {
        "performed": False,
        "status": "not_needed",
        "verdicts": [],
    }
    assert all(item["stage"] != "vocabulary_review" for item in trace["model_diagnostics"])
