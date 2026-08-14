import pytest

from paperless_clerk.config import Settings
from paperless_clerk.metadata import MetadataPlanner, apply_metadata_plan
from paperless_clerk.schemas import MetadataChoice, MetadataProposal


class FakePaperless:
    def __init__(self):
        self.patch = None

    async def ensure_entity(self, resource: str, name: str) -> dict:
        assert resource == "tags"
        assert name == "Aetna"
        return {"id": 99, "name": name}

    async def update_document(self, document_id: int, patch: dict) -> dict:
        self.patch = patch
        return {"id": document_id, **patch, "modified": "2026-08-13T12:00:00Z"}


def metadata_choice(*, existing_id: int | None = None, new_name: str | None = None):
    return MetadataChoice(
        existing_id=existing_id,
        new_name=new_name,
        confidence=0.95,
        reason="Useful reusable filing concept",
        evidence="Aetna medical coverage",
        source_pages=[1],
    )


@pytest.mark.asyncio
async def test_reused_and_created_tags_are_applied_together() -> None:
    catalog = {
        "tags": [{"id": 1, "name": "Medical"}],
        "correspondents": [],
        "document_types": [],
        "custom_fields": [],
    }
    proposal = MetadataProposal(
        tags=[metadata_choice(existing_id=1), metadata_choice(new_name="Aetna")]
    )
    settings = Settings()
    plan = MetadataPlanner(settings).validate(proposal, catalog)
    paperless = FakePaperless()

    applied, _ = await apply_metadata_plan(
        plan=plan,
        document={"id": 8, "title": "Policy", "tags": [1], "custom_fields": []},
        paperless=paperless,  # type: ignore[arg-type]
        settings=settings,
    )

    assert paperless.patch == {"tags": [1, 99]}
    assert applied["created"][0] == {
        "resource": "tags",
        "id": 99,
        "name": "Aetna",
        "confidence": 0.95,
        "reason": "Useful reusable filing concept",
        "evidence": "Aetna medical coverage",
        "source_pages": [1],
    }
    assert applied["reused"][0]["name"] == "Medical"


@pytest.mark.asyncio
async def test_watch_tag_is_consumed_only_in_successful_metadata_patch() -> None:
    settings = Settings(automation_tag="Clerk Inbox")
    plan = MetadataPlanner(settings).validate(
        MetadataProposal(),
        {"tags": [], "correspondents": [], "document_types": [], "custom_fields": []},
    )
    paperless = FakePaperless()

    applied, _ = await apply_metadata_plan(
        plan=plan,
        document={"id": 9, "title": "Agreement", "tags": [7, 55], "custom_fields": []},
        paperless=paperless,  # type: ignore[arg-type]
        settings=settings,
        consume_tag={"id": 55, "name": "Clerk Inbox"},
    )

    assert paperless.patch == {"tags": [7]}
    assert applied["removed"] == [
        {
            "resource": "tags",
            "id": 55,
            "name": "Clerk Inbox",
            "reason": "automation watch tag consumed after successful metadata processing",
        }
    ]
