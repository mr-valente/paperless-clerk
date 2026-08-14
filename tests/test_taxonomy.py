from paperless_clerk.config import Settings
from paperless_clerk.domain.taxonomy import Entity, find_duplicate
from paperless_clerk.metadata import MetadataPlanner
from paperless_clerk.schemas import MetadataChoice, MetadataProposal


def catalog() -> dict:
    return {
        "tags": [
            {"id": 1, "name": "Medical", "document_count": 42},
            {"id": 2, "name": "Taxes", "document_count": 31},
            {"id": 3, "name": "Insurance", "document_count": 19},
        ],
        "correspondents": [{"id": 10, "name": "IRS", "document_count": 12}],
        "document_types": [{"id": 20, "name": "Invoice", "document_count": 50}],
        "custom_fields": [],
    }


def choice(*, existing_id: int | None = None, new_name: str | None = None) -> MetadataChoice:
    return MetadataChoice(
        existing_id=existing_id,
        new_name=new_name,
        confidence=0.94,
        reason="Strong reusable concept supported by the document",
        evidence="short evidence",
        source_pages=[1],
    )


def test_correspondent_acronym_is_semantically_reused() -> None:
    match = find_duplicate("Internal Revenue Service", [Entity(10, "IRS")], "correspondent")

    assert match is not None
    assert match.entity.id == 10
    assert match.reason == "acronym of existing name"


def test_near_duplicate_tag_is_prevented() -> None:
    proposal = MetadataProposal(tags=[choice(new_name="Tax Documents")])

    plan = MetadataPlanner(Settings()).validate(proposal, catalog())

    assert plan.tags[0]["action"] == "reuse"
    assert plan.tags[0]["id"] == 2
    assert plan.tags[0]["name"] == "Taxes"
    assert plan.duplicate_candidates[0]["proposed"] == "Tax Documents"


def test_modifier_only_tag_reuses_broader_existing_concept() -> None:
    proposal = MetadataProposal(tags=[choice(new_name="Income Taxes")])

    plan = MetadataPlanner(Settings()).validate(proposal, catalog())

    assert plan.tags[0]["action"] == "reuse"
    assert plan.tags[0]["name"] == "Taxes"
    assert plan.duplicate_candidates[0]["reason"] == "existing concept with only a modifier"


def test_modifier_tag_can_grow_vocabulary_with_explicit_distinction() -> None:
    proposed = MetadataChoice(
        new_name="Income Taxes",
        confidence=0.96,
        reason=(
            "Income Taxes is distinct from Taxes because this library separately retrieves "
            "income-tax filings from property-tax records."
        ),
        evidence="Form 1099-INT",
        source_pages=[1],
    )

    plan = MetadataPlanner(Settings()).validate(
        MetadataProposal(tags=[choice(existing_id=2), proposed]), catalog()
    )

    assert {(item["action"], item["name"]) for item in plan.tags} == {
        ("reuse", "Taxes"),
        ("create", "Income Taxes"),
    }
    assert plan.duplicate_candidates[0]["outcome"] == "created_as_explicit_distinction"


def test_existing_and_new_tags_can_coexist() -> None:
    proposal = MetadataProposal(
        tags=[choice(existing_id=1), choice(existing_id=3), choice(new_name="Aetna")]
    )

    plan = MetadataPlanner(Settings()).validate(proposal, catalog())

    assert {(item["action"], item["name"]) for item in plan.tags} == {
        ("reuse", "Medical"),
        ("reuse", "Insurance"),
        ("create", "Aetna"),
    }


def test_model_generated_existing_id_is_validated_by_resource_type() -> None:
    proposal = MetadataProposal(
        correspondent=choice(existing_id=20),  # This is a document-type ID, not a correspondent.
        tags=[choice(existing_id=9999)],
    )

    plan = MetadataPlanner(Settings()).validate(proposal, catalog())

    assert plan.correspondent is None
    assert plan.tags == []
    assert {item["reason"] for item in plan.rejected} == {"unknown or wrong-type existing ID"}


def test_low_confidence_new_vocabulary_is_omitted() -> None:
    low = MetadataChoice(
        new_name="One-off phrase",
        confidence=0.2,
        reason="Maybe useful",
        evidence="unclear",
    )
    plan = MetadataPlanner(Settings()).validate(MetadataProposal(tags=[low]), catalog())

    assert not plan.tags
    assert plan.rejected[0]["reason"] == "below confidence threshold"


def test_duplicate_new_values_within_one_proposal_are_collapsed() -> None:
    proposal = MetadataProposal(
        tags=[choice(new_name="Annual Taxes"), choice(new_name="Annual Tax Documents")]
    )

    plan = MetadataPlanner(Settings()).validate(proposal, catalog())

    assert len(plan.tags) == 1
    assert plan.duplicate_candidates[-1]["reason"].startswith("duplicate within this proposal")


def test_watch_tag_cannot_be_proposed_as_document_metadata() -> None:
    plan = MetadataPlanner(Settings(automation_tag="Clerk Inbox")).validate(
        MetadataProposal(tags=[choice(new_name=" clerk   inbox ")]), catalog()
    )

    assert plan.tags == []
    assert plan.rejected[0]["reason"] == "automation watch tag is workflow-only"
