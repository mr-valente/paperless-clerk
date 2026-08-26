import json
from pathlib import Path

import pytest

from paperless_clerk.config import Settings, SettingsManager, load_persisted_settings
from paperless_clerk.db import Database
from paperless_clerk.ocr_profiles import VLLM_FLAG, ocr_profile


def test_omitted_secret_is_preserved_and_explicit_blank_clears_it(tmp_path: Path) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    manager = SettingsManager(db)
    manager.update({"paperless_token": "initial-secret"})

    updated = manager.update({"paperless_url": "http://paperless:8000"})

    assert updated.paperless_token.get_secret_value() == "initial-secret"
    assert "initial-secret" not in str(updated.public_dict())
    assert updated.public_dict()["paperless_token_configured"] is True

    cleared = manager.update({"paperless_token": ""})

    assert cleared.paperless_token.get_secret_value() == ""
    assert cleared.public_dict()["paperless_token_configured"] is False


def test_output_budget_must_fit_model_context() -> None:
    with pytest.raises(ValueError, match="metadata maximum output"):
        Settings(metadata_context_tokens=4096, metadata_max_output_tokens=4096)


def test_ocr_profile_defaults_to_generic_and_accepts_specialists() -> None:
    assert Settings().ocr_profile == "generic"
    assert Settings(ocr_profile="deepseek_ocr").ocr_profile == "deepseek_ocr"
    assert Settings(ocr_profile="deepseek_ocr_llamacpp").ocr_profile == "deepseek_ocr_llamacpp"
    assert Settings(ocr_profile="glm_ocr").ocr_profile == "glm_ocr"
    with pytest.raises(ValueError, match="ocr_profile"):
        Settings(ocr_profile="some-other-model")


def test_original_ocr_version_retention_defaults_on_and_can_be_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().keep_original_version is True
    assert Settings(keep_original_version=False).keep_original_version is False

    db = Database(tmp_path / "clerk.db")
    db.initialize()
    manager = SettingsManager(db)
    manager.update({"keep_original_version": False})

    assert SettingsManager(db).get().keep_original_version is False

    monkeypatch.setenv("CLERK_KEEP_ORIGINAL_VERSION", "true")
    environment_managed = SettingsManager(db).get()

    assert environment_managed.keep_original_version is True
    assert "keep_original_version" in environment_managed.public_dict()["environment_overrides"]


def test_held_back_vllm_profiles_are_offered_only_behind_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offered = [choice["key"] for choice in Settings().public_dict()["ocr_profile_choices"]]
    assert offered == ["generic", "deepseek_ocr_llamacpp"]

    monkeypatch.setenv(VLLM_FLAG, "true")

    offered = [choice["key"] for choice in Settings().public_dict()["ocr_profile_choices"]]
    assert offered == ["generic", "deepseek_ocr", "deepseek_ocr_llamacpp", "glm_ocr"]


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_a_held_back_profile_resolves_to_generic_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # A database that still names a vLLM profile has to keep loading, so the
    # request contract degrades rather than the container refusing to start.
    monkeypatch.setenv(VLLM_FLAG, value)

    assert Settings(ocr_profile="glm_ocr").ocr_profile == "glm_ocr"
    assert ocr_profile("glm_ocr").key == "generic"
    assert ocr_profile("deepseek_ocr").key == "generic"
    # Neither of the working paths is gated.
    assert ocr_profile("deepseek_ocr_llamacpp").key == "deepseek_ocr_llamacpp"
    assert ocr_profile("generic").key == "generic"


def test_the_flag_restores_the_vllm_request_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VLLM_FLAG, "1")

    assert ocr_profile("glm_ocr").key == "glm_ocr"
    assert "vllm_xargs" in ocr_profile("deepseek_ocr").extra_body


def test_explicit_environment_value_overrides_persisted_ui_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    first = SettingsManager(db)
    first.update({"paperless_url": "http://saved-paperless:8000"})
    monkeypatch.setenv("PAPERLESS_URL", "http://environment-paperless:8000")

    restarted = SettingsManager(db)

    assert restarted.get().paperless_url == "http://environment-paperless:8000"
    assert (
        restarted.update({"paperless_url": "http://ignored-ui-value:8000"}).paperless_url
        == "http://environment-paperless:8000"
    )


def test_shared_openai_environment_url_is_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    manager = SettingsManager(db)
    manager.update({"openai_base_url": "http://saved-model-server:11434/v1"})
    monkeypatch.setenv("CLERK_OPENAI_BASE_URL", "http://environment-model-server:1234/v1/")

    restarted = SettingsManager(db)

    assert restarted.get().openai_base_url == "http://environment-model-server:1234/v1"
    assert "openai_base_url" in restarted.get().public_dict()["environment_overrides"]


def test_ocr_profile_can_be_persisted_or_managed_by_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "clerk.db")
    db.initialize()
    manager = SettingsManager(db)

    assert manager.update({"ocr_profile": "deepseek_ocr"}).ocr_profile == "deepseek_ocr"

    monkeypatch.setenv("CLERK_OCR_PROFILE", "generic")
    restarted = SettingsManager(db)

    assert restarted.get().ocr_profile == "generic"
    assert "ocr_profile" in restarted.get().public_dict()["environment_overrides"]


def test_saved_glm_profile_is_restored() -> None:
    settings = load_persisted_settings(json.dumps({"ocr_profile": "glm_ocr"}))

    assert settings.ocr_profile == "glm_ocr"


def test_legacy_model_urls_migrate_to_one_shared_url() -> None:
    settings = load_persisted_settings(
        json.dumps(
            {
                "ocr_base_url": "http://vision-server:11434/v1",
                "metadata_base_url": "http://text-server:11434/v1",
                "ocr_api_key": "legacy-key",
            }
        )
    )

    assert settings.openai_base_url == "http://vision-server:11434/v1"
    assert settings.openai_api_key.get_secret_value() == "legacy-key"
    assert "ocr_base_url" not in settings.persisted_dict()
    assert "metadata_base_url" not in settings.persisted_dict()
    assert "ocr_api_key" not in settings.persisted_dict()
    assert "metadata_api_key" not in settings.persisted_dict()


def test_ntfy_settings_require_a_valid_topic_and_keep_token_private() -> None:
    with pytest.raises(ValueError, match="topic is required"):
        Settings(notifications_enabled=True)
    with pytest.raises(ValueError, match="may contain only"):
        Settings(ntfy_topic="invalid topic/name")

    settings = Settings(
        notifications_enabled=True,
        ntfy_topic="clerk_private-alerts",
        ntfy_token="secret-token",
    )

    assert settings.persisted_dict()["ntfy_token"] == "secret-token"
    assert "ntfy_token" not in settings.public_dict()
    assert settings.public_dict()["ntfy_token_configured"] is True
    assert "secret-token" not in str(settings.public_dict())
