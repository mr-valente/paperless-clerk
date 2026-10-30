from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnqueueRequest(BaseModel):
    document_ids: list[int] = Field(min_length=1, max_length=500)
    mode: Literal["full", "ocr", "metadata"] = "full"
    # Per-job override of the app-wide setting; omitted means follow the setting.
    keep_original_version: bool | None = None

    @model_validator(mode="after")
    def unique_positive_ids(self) -> EnqueueRequest:
        self.document_ids = list(dict.fromkeys(self.document_ids))
        if any(document_id < 1 for document_id in self.document_ids):
            raise ValueError("document IDs must be positive")
        return self


class ResolveConflictRequest(BaseModel):
    resolution: Literal["keep_existing", "use_clerk"]


class ClearJobsRequest(BaseModel):
    """Statuses whose finished jobs may be deleted. Completed and needs-review jobs
    carry decisions or conflicts, so only exhausted and cancelled runs qualify."""

    statuses: list[Literal["failed", "cancelled"]] = Field(
        default_factory=lambda: ["failed", "cancelled"], min_length=1
    )


class SettingsPatch(BaseModel):
    values: dict[str, Any]


class MetadataChoice(StrictModel):
    existing_id: int | None = None
    new_name: str | None = None
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    evidence: str = Field(default="", max_length=500)
    source_pages: list[int] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def exactly_one_target(self) -> MetadataChoice:
        if (self.existing_id is None) == (not self.new_name):
            raise ValueError("provide exactly one of existing_id or new_name")
        if self.new_name:
            self.new_name = " ".join(self.new_name.split())[:100]
        self.source_pages = sorted(set(self.source_pages))
        return self


class ScalarCandidate(StrictModel):
    value: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    evidence: str = Field(default="", max_length=500)
    source_pages: list[int] = Field(default_factory=list, max_length=30)

    @field_validator("value")
    @classmethod
    def normalize_value(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("value may not be blank")
        return normalized


class CustomFieldCandidate(StrictModel):
    field_id: int
    value: Any
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    evidence: str = Field(default="", max_length=500)
    source_pages: list[int] = Field(default_factory=list, max_length=30)


class NewCustomFieldCandidate(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    data_type: Literal[
        "string", "longtext", "boolean", "integer", "float", "monetary", "date", "url"
    ]
    value: Any
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=500)
    evidence: str = Field(default="", max_length=500)
    source_pages: list[int] = Field(default_factory=list, max_length=30)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("name may not be blank")
        return normalized


class ChunkAnalysis(StrictModel):
    correspondent_candidates: list[MetadataChoice] = Field(default_factory=list, max_length=4)
    document_type_candidates: list[MetadataChoice] = Field(default_factory=list, max_length=4)
    tag_candidates: list[MetadataChoice] = Field(default_factory=list, max_length=12)
    title_candidates: list[ScalarCandidate] = Field(default_factory=list, max_length=4)
    date_candidates: list[ScalarCandidate] = Field(default_factory=list, max_length=6)
    custom_field_candidates: list[CustomFieldCandidate] = Field(default_factory=list, max_length=20)
    new_custom_field_candidates: list[NewCustomFieldCandidate] = Field(
        default_factory=list, max_length=8
    )


class MetadataProposal(StrictModel):
    correspondent: MetadataChoice | None = None
    document_type: MetadataChoice | None = None
    tags: list[MetadataChoice] = Field(default_factory=list, max_length=20)
    title: ScalarCandidate | None = None
    document_date: ScalarCandidate | None = None
    custom_fields: list[CustomFieldCandidate] = Field(default_factory=list, max_length=30)
    new_custom_fields: list[NewCustomFieldCandidate] = Field(default_factory=list, max_length=10)
    summary: str = Field(default="", max_length=700)


VocabularyKind = Literal["tag", "correspondent", "document_type"]


class VocabularyVerdict(StrictModel):
    """The gatekeeper's answer for one proposed new vocabulary entry."""

    kind: VocabularyKind
    proposed_name: str = Field(min_length=1, max_length=160)
    decision: Literal["create", "reuse", "drop"]
    existing_id: int | None = None
    name: str | None = Field(default=None, max_length=100)
    reason: str = Field(min_length=1, max_length=400)


class VocabularyReview(StrictModel):
    verdicts: list[VocabularyVerdict] = Field(default_factory=list, max_length=24)


class VocabularyMergeSuggestion(StrictModel):
    kind: VocabularyKind
    keep_id: int
    merge_id: int
    reason: str = Field(min_length=1, max_length=400)


class VocabularyAudit(StrictModel):
    suggestions: list[VocabularyMergeSuggestion] = Field(default_factory=list, max_length=40)


class VocabularyMergeRequest(BaseModel):
    kind: VocabularyKind
    keep_id: int = Field(ge=1)
    merge_id: int = Field(ge=1)

    @model_validator(mode="after")
    def distinct_entries(self) -> VocabularyMergeRequest:
        if self.keep_id == self.merge_id:
            raise ValueError("keep_id and merge_id must differ")
        return self


class VocabularyDismissRequest(BaseModel):
    kind: VocabularyKind
    keep_id: int = Field(ge=1)
    merge_id: int = Field(ge=1)


class TagReview(StrictModel):
    """A focused second opinion when the general metadata pass found no usable tags."""

    tags: list[MetadataChoice] = Field(max_length=12)
    assessment: str = Field(min_length=1, max_length=700)

    @field_validator("assessment", mode="before")
    @classmethod
    def bound_assessment(cls, value: Any) -> Any:
        """Keep a verbose explanation from discarding otherwise valid tag choices."""
        if not isinstance(value, str):
            return value
        normalized = " ".join(value.split())
        if len(normalized) <= 700:
            return normalized
        return f"{normalized[:697].rstrip()}..."


def _allows_null(property_schema: dict[str, Any]) -> bool:
    if property_schema.get("type") == "null":
        return True
    options = property_schema.get("anyOf") or property_schema.get("oneOf") or []
    return any(isinstance(option, dict) and option.get("type") == "null" for option in options)


def model_prompt_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON Schema sent to the model for structured output.

    Pydantic leaves nullable fields optional, so a grammar-constrained server such as
    llama.cpp lets the model skip them. Local models then emit every required property
    and never return to ``existing_id``/``new_name``, which breaks the "exactly one target"
    rule. Requiring nullable properties makes an explicit ``null`` the only way to leave
    a target empty, so omission can no longer be mistaken for absence. Parsing still
    uses the pydantic model, so explicit nulls validate unchanged.
    """
    schema = copy.deepcopy(model.model_json_schema())
    objects = [schema, *schema.get("$defs", {}).values()]
    for node in objects:
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        required = list(node.get("required", []))
        for name, property_schema in properties.items():
            if (
                name not in required
                and isinstance(property_schema, dict)
                and _allows_null(property_schema)
            ):
                required.append(name)
        if required:
            node["required"] = required
    return schema
