from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from paperless_clerk.ocr_profiles import PROFILE_KEYS, available_profiles


class Settings(BaseModel):
    """Runtime settings.

    Values changed in the UI are persisted in SQLite. Explicit environment
    values remain authoritative, and secret values are never returned by the
    settings API.
    """

    paperless_url: str = "http://paperless-webserver:8000"
    paperless_token: SecretStr = SecretStr("")
    paperless_verify_ssl: bool = True

    openai_base_url: str = "http://host.docker.internal:11434/v1"
    openai_api_key: SecretStr = SecretStr("")
    ocr_model: str = "qwen2.5vl:7b"
    ocr_profile: str = "generic"
    prefer_clerk_ocr: bool = True
    # No OCR context setting: one page image plus a short command is the whole
    # request, so the server's own context is the only limit that matters and
    # it already clamps an oversized output request.
    ocr_max_output_tokens: int = Field(default=4096, ge=256, le=131_072)

    metadata_model: str = "qwen2.5:14b"
    # Unlike OCR, this one is load-bearing: it sizes the chunk budget below.
    metadata_context_tokens: int = Field(default=16384, ge=2048, le=1_000_000)
    metadata_max_output_tokens: int = Field(default=4096, ge=256, le=131_072)

    request_timeout_seconds: int = Field(default=300, ge=10, le=3600)
    model_max_retries: int = Field(default=3, ge=0, le=10)
    job_max_attempts: int = Field(default=3, ge=1, le=10)
    job_workers: int = Field(default=2, ge=1, le=8)
    page_concurrency: int = Field(default=1, ge=1, le=16)
    metadata_concurrency: int = Field(default=1, ge=1, le=8)
    lease_seconds: int = Field(default=900, ge=60, le=7200)

    render_dpi: int = Field(default=160, ge=72, le=400)
    max_image_pixels: int = Field(default=16_000_000, ge=1_000_000, le=80_000_000)
    jpeg_quality: int = Field(default=86, ge=45, le=95)
    ocr_min_chars: int = Field(default=24, ge=1, le=1000)
    ocr_similarity_threshold: float = Field(default=0.82, ge=0.5, le=0.99)
    conflict_tag: str = Field(default="ocr-conflict", min_length=1, max_length=100)

    metadata_chunk_chars: int = Field(default=12_000, ge=2_000, le=100_000)
    metadata_candidate_limit: int = Field(default=80, ge=10, le=500)
    metadata_reduce_batch_size: int = Field(default=16, ge=4, le=50)
    metadata_min_confidence: float = Field(default=0.68, ge=0.0, le=1.0)
    metadata_apply_mode: Literal["missing_only", "overwrite"] = "missing_only"
    allow_new_tags: bool = True
    allow_new_correspondents: bool = True
    allow_new_document_types: bool = True
    allow_new_custom_fields: bool = False

    automation_enabled: bool = False
    automation_interval_seconds: int = Field(default=120, ge=15, le=86_400)
    automation_page_size: int = Field(default=25, ge=1, le=100)
    automation_tag: str = Field(default="", max_length=100)

    notifications_enabled: bool = False
    ntfy_url: str = "https://ntfy.sh"
    ntfy_topic: str = Field(default="", max_length=64)
    ntfy_token: SecretStr = SecretStr("")

    appearance_theme: Literal["system", "light", "dark"] = "system"
    appearance_density: Literal["comfortable", "compact"] = "comfortable"
    appearance_motion: Literal["system", "full", "reduced"] = "system"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("paperless_url", "openai_base_url", "ntfy_url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value

    @field_validator("ocr_profile")
    @classmethod
    def known_ocr_profile(cls, value: str) -> str:
        if value not in PROFILE_KEYS:
            raise ValueError(f"must be one of: {', '.join(PROFILE_KEYS)}")
        return value

    @field_validator("conflict_tag", "automation_tag")
    @classmethod
    def normalize_tag(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("ntfy_topic")
    @classmethod
    def validate_ntfy_topic(cls, value: str) -> str:
        value = value.strip()
        if value and any(
            not (character.isascii() and character.isalnum()) and character not in "-_"
            for character in value
        ):
            raise ValueError(
                "ntfy topic may contain only letters, numbers, dashes, and underscores"
            )
        return value

    @model_validator(mode="after")
    def keep_chunks_inside_context(self) -> Settings:
        if self.automation_tag and self.automation_tag.casefold() == self.conflict_tag.casefold():
            raise ValueError("automation watch tag must differ from the OCR conflict tag")
        if self.notifications_enabled and not self.ntfy_topic:
            raise ValueError("an ntfy topic is required when notifications are enabled")
        if self.metadata_max_output_tokens >= self.metadata_context_tokens:
            raise ValueError(
                "metadata maximum output tokens must be smaller than its context limit"
            )
        if self.metadata_context_tokens - self.metadata_max_output_tokens < 2_048:
            raise ValueError("metadata context must reserve at least 2048 input tokens")
        # Three characters per token leaves headroom for noisy and multilingual OCR.
        maximum = max(
            2_000,
            (self.metadata_context_tokens - self.metadata_max_output_tokens - 1_000) * 3,
        )
        if self.metadata_chunk_chars > maximum:
            self.metadata_chunk_chars = maximum
        return self

    def secret_value(self, name: str) -> str:
        value = getattr(self, name)
        return value.get_secret_value() if isinstance(value, SecretStr) else str(value)

    def persisted_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["paperless_token"] = self.paperless_token.get_secret_value()
        data["openai_api_key"] = self.openai_api_key.get_secret_value()
        data["ntfy_token"] = self.ntfy_token.get_secret_value()
        return data

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump(
            mode="json", exclude={"paperless_token", "openai_api_key", "ntfy_token"}
        )
        data.update(
            {
                "paperless_token_configured": bool(self.paperless_token.get_secret_value()),
                "openai_api_key_configured": bool(self.openai_api_key.get_secret_value()),
                "ntfy_token_configured": bool(self.ntfy_token.get_secret_value()),
                # Only the profiles on offer. A held-back one stays valid so a
                # database that names it still loads; the UI just stops showing it.
                "ocr_profile_choices": [
                    {"key": profile.key, "label": profile.label} for profile in available_profiles()
                ],
                # Names only: this lets the UI explain why a container-managed
                # value is read-only without exposing any environment secrets.
                "environment_overrides": sorted(environment_values()),
            }
        )
        return data


ENVIRONMENT_FIELDS = {
    "PAPERLESS_URL": "paperless_url",
    "PAPERLESS_TOKEN": "paperless_token",
    "PAPERLESS_VERIFY_SSL": "paperless_verify_ssl",
    "CLERK_OPENAI_BASE_URL": "openai_base_url",
    "CLERK_OPENAI_API_KEY": "openai_api_key",
    "CLERK_OCR_MODEL": "ocr_model",
    "CLERK_OCR_PROFILE": "ocr_profile",
    "CLERK_PREFER_CLERK_OCR": "prefer_clerk_ocr",
    "CLERK_OCR_MAX_OUTPUT_TOKENS": "ocr_max_output_tokens",
    "CLERK_METADATA_MODEL": "metadata_model",
    "CLERK_METADATA_CONTEXT_TOKENS": "metadata_context_tokens",
    "CLERK_METADATA_MAX_OUTPUT_TOKENS": "metadata_max_output_tokens",
    "CLERK_REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
    "CLERK_MODEL_MAX_RETRIES": "model_max_retries",
    "CLERK_JOB_MAX_ATTEMPTS": "job_max_attempts",
    "CLERK_JOB_WORKERS": "job_workers",
    "CLERK_PAGE_CONCURRENCY": "page_concurrency",
    "CLERK_METADATA_CONCURRENCY": "metadata_concurrency",
    "CLERK_RENDER_DPI": "render_dpi",
    "CLERK_MAX_IMAGE_PIXELS": "max_image_pixels",
    "CLERK_JPEG_QUALITY": "jpeg_quality",
    "CLERK_OCR_MIN_CHARS": "ocr_min_chars",
    "CLERK_OCR_SIMILARITY_THRESHOLD": "ocr_similarity_threshold",
    "CLERK_CONFLICT_TAG": "conflict_tag",
    "CLERK_METADATA_CHUNK_CHARS": "metadata_chunk_chars",
    "CLERK_METADATA_CANDIDATE_LIMIT": "metadata_candidate_limit",
    "CLERK_METADATA_MIN_CONFIDENCE": "metadata_min_confidence",
    "CLERK_METADATA_APPLY_MODE": "metadata_apply_mode",
    "CLERK_ALLOW_NEW_TAGS": "allow_new_tags",
    "CLERK_ALLOW_NEW_CORRESPONDENTS": "allow_new_correspondents",
    "CLERK_ALLOW_NEW_DOCUMENT_TYPES": "allow_new_document_types",
    "CLERK_ALLOW_NEW_CUSTOM_FIELDS": "allow_new_custom_fields",
    "CLERK_AUTOMATION_ENABLED": "automation_enabled",
    "CLERK_AUTOMATION_INTERVAL_SECONDS": "automation_interval_seconds",
    "CLERK_AUTOMATION_PAGE_SIZE": "automation_page_size",
    "CLERK_AUTOMATION_TAG": "automation_tag",
    "CLERK_NOTIFICATIONS_ENABLED": "notifications_enabled",
    "CLERK_NTFY_URL": "ntfy_url",
    "CLERK_NTFY_TOPIC": "ntfy_topic",
    "CLERK_NTFY_TOKEN": "ntfy_token",
    "CLERK_LOG_LEVEL": "log_level",
}


def environment_values() -> dict[str, str]:
    return {
        field: os.environ[name]
        for name, field in ENVIRONMENT_FIELDS.items()
        if name in os.environ and os.environ[name] != ""
    }


def settings_from_environment() -> Settings:
    return Settings.model_validate(environment_values())


def data_directory() -> Path:
    return Path(os.environ.get("CLERK_DATA_DIR", "./data")).expanduser().resolve()


def load_persisted_settings(raw_json: str | None) -> Settings:
    if not raw_json:
        return settings_from_environment()
    raw = json.loads(raw_json)
    # Pre-0.1 databases stored one URL per model purpose. Prefer the OCR URL
    # when they differed, because OCR is the first model stage, then discard
    # both legacy keys so future saves contain only the unified setting.
    legacy_base_url = raw.get("ocr_base_url") or raw.get("metadata_base_url")
    if "openai_base_url" not in raw and legacy_base_url:
        raw["openai_base_url"] = legacy_base_url
    raw.pop("ocr_base_url", None)
    raw.pop("metadata_base_url", None)
    legacy_api_key = raw.get("ocr_api_key") or raw.get("metadata_api_key")
    if "openai_api_key" not in raw and legacy_api_key:
        raw["openai_api_key"] = legacy_api_key
    raw.pop("ocr_api_key", None)
    raw.pop("metadata_api_key", None)
    # Explicit container environment remains authoritative on restart. Values
    # absent from the environment continue to use UI-persisted configuration.
    raw.update(environment_values())
    return Settings.model_validate(raw)


class SettingsManager:
    def __init__(self, database: Any):
        self.database = database
        self._lock = threading.RLock()
        persisted = database.get_setting("runtime")
        self._settings = load_persisted_settings(persisted)
        if persisted is None:
            database.set_setting("runtime", json.dumps(self._settings.persisted_dict()))

    def get(self) -> Settings:
        with self._lock:
            return self._settings.model_copy(deep=True)

    def update(self, values: dict[str, Any]) -> Settings:
        unknown = set(values) - set(Settings.model_fields)
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        with self._lock:
            merged = self._settings.persisted_dict()
            for key, value in values.items():
                merged[key] = value
            # Container-managed values stay authoritative for the lifetime of
            # the process as well as after a restart.
            merged.update(environment_values())
            updated = Settings.model_validate(merged)
            self.database.set_setting("runtime", json.dumps(updated.persisted_dict()))
            self._settings = updated
            return updated.model_copy(deep=True)
