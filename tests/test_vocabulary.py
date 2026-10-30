from __future__ import annotations

import pytest

from paperless_clerk.clients.openai_compatible import ModelError
from paperless_clerk.config import Settings
from paperless_clerk.vocabulary import (
    lexical_suggestions,
    merge_entities,
    pair_key,
    review_vocabulary,
    visible_catalog,
)

CATALOG = {
    "tags": [
        {"id": 1, "name": "Taxes", "document_count": 6},
        {"id": 2, "name": "Income Taxes", "document_count": 1},
        {"id": 3, "name": "Optical", "document_count": 2},
        {"id": 4, "name": "Optometry", "document_count": 1},
        {"id": 5, "name": "Inbox", "document_count": 0, "is_inbox_tag": True},
        {"id": 6, "name": "paperless-gpt-failed", "document_count": 0},
    ],
    "correspondents": [
        {"id": 10, "name": "New Jersey Motor Vehicle Commission", "document_count": 1},
        {"id": 11, "name": "NJMVC", "document_count": 0},
    ],
    "document_types": [{"id": 20, "name": "Invoice", "document_count": 3}],
    "custom_fields": [],
}


def test_visible_catalog_hides_inbox_and_configured_workflow_tags() -> None:
    settings = Settings(metadata_hidden_tags="paperless-gpt-failed")

    visible = visible_catalog(CATALOG, settings)

    assert [tag["id"] for tag in visible["tags"]] == [1, 2, 3, 4]
    assert visible["correspondents"] == CATALOG["correspondents"]


def test_lexical_suggestions_keep_the_busier_or_spelled_out_entry() -> None:
    suggestions = lexical_suggestions(visible_catalog(CATALOG, Settings()))

    pairs = {(item["kind"], item["keep_name"], item["merge_name"]): item for item in suggestions}
    assert ("tag", "Taxes", "Income Taxes") in pairs
    assert ("correspondent", "New Jersey Motor Vehicle Commission", "NJMVC") in pairs
    # Optical and Optometry share no tokens; only the model pass can pair them.
    assert not any(
        {item["keep_name"], item["merge_name"]} == {"Optical", "Optometry"} for item in suggestions
    )
    assert pairs[("tag", "Taxes", "Income Taxes")]["source"] == "lexical"
    assert pairs[("tag", "Taxes", "Income Taxes")]["key"] == pair_key("tag", 2, 1)


class FakePaperless:
    def __init__(self, catalog: dict):
        self.catalog_data = catalog
        self.bulk_edits: list[tuple[list[int], str, dict]] = []
        self.deleted: list[tuple[str, int]] = []
        self.documents = {("tag", 4): [101, 102], ("correspondent", 11): [103]}

    async def catalog(self) -> dict:
        return self.catalog_data

    async def list_resource(self, resource: str, *, params: dict | None = None) -> list[dict]:
        return self.catalog_data[resource]

    async def document_ids_with(self, kind: str, entity_id: int) -> list[int]:
        return self.documents.get((kind, entity_id), [])

    async def bulk_edit(self, document_ids: list[int], method: str, parameters: dict) -> None:
        self.bulk_edits.append((document_ids, method, parameters))

    async def delete_entity(self, resource: str, entity_id: int) -> None:
        self.deleted.append((resource, entity_id))


class FakeAuditModel:
    def __init__(self, response: dict | None = None, fail: bool = False):
        self.response = response
        self.fail = fail
        self.calls: list[str] = []

    async def structured(self, *, name: str, schema: dict, system: str, user: str) -> dict:
        self.calls.append(name)
        if self.fail:
            raise ModelError("model offline", retryable=False)
        return self.response or {"suggestions": []}


@pytest.mark.asyncio
async def test_review_merges_lexical_and_validated_model_suggestions() -> None:
    model = FakeAuditModel(
        {
            "suggestions": [
                {
                    "kind": "tag",
                    "keep_id": 3,
                    "merge_id": 4,
                    "reason": "Both file eye-care documents",
                },
                {"kind": "tag", "keep_id": 1, "merge_id": 2, "reason": "Already found lexically"},
                {"kind": "tag", "keep_id": 3, "merge_id": 999, "reason": "Unknown entry"},
                {"kind": "document_type", "keep_id": 20, "merge_id": 3, "reason": "Wrong kind"},
            ]
        }
    )

    review = await review_vocabulary(Settings(), FakePaperless(CATALOG), model)  # type: ignore[arg-type]

    assert model.calls == ["clerk_vocabulary_audit"]
    assert review["model_status"] == "completed"
    assert review["counts"] == {"tags": 5, "document_types": 1, "correspondents": 2}
    described = [
        (item["kind"], item["keep_name"], item["merge_name"], item["source"])
        for item in review["suggestions"]
    ]
    assert described == [
        ("tag", "Optical", "Optometry", "model"),
        ("tag", "Taxes", "Income Taxes", "lexical"),
        ("correspondent", "New Jersey Motor Vehicle Commission", "NJMVC", "lexical"),
    ]


@pytest.mark.asyncio
async def test_review_survives_a_failed_model_pass() -> None:
    review = await review_vocabulary(Settings(), FakePaperless(CATALOG), FakeAuditModel(fail=True))  # type: ignore[arg-type]

    assert review["model_status"].startswith("failed:")
    assert [item["source"] for item in review["suggestions"]] == ["lexical", "lexical"]


@pytest.mark.asyncio
async def test_merge_moves_documents_before_deleting_the_emptied_entry() -> None:
    paperless = FakePaperless(CATALOG)

    tag_result = await merge_entities(paperless, "tag", 3, 4)  # type: ignore[arg-type]
    correspondent_result = await merge_entities(paperless, "correspondent", 10, 11)  # type: ignore[arg-type]

    assert paperless.bulk_edits == [
        ([101, 102], "modify_tags", {"add_tags": [3], "remove_tags": [4]}),
        ([103], "set_correspondent", {"correspondent": 10}),
    ]
    assert paperless.deleted == [("tags", 4), ("correspondents", 11)]
    assert tag_result == {
        "kind": "tag",
        "keep": {"id": 3, "name": "Optical"},
        "merged": {"id": 4, "name": "Optometry"},
        "documents": 2,
    }
    assert correspondent_result["documents"] == 1


@pytest.mark.asyncio
async def test_merge_refuses_entries_that_no_longer_exist() -> None:
    paperless = FakePaperless(CATALOG)

    with pytest.raises(LookupError):
        await merge_entities(paperless, "tag", 3, 404)  # type: ignore[arg-type]

    assert paperless.bulk_edits == [] and paperless.deleted == []
