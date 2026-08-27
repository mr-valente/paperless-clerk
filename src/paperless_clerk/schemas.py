from __future__ import annotations

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
