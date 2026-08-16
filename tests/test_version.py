import pytest

from paperless_clerk import _installed_version, _runtime_version


def test_runtime_version_uses_container_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPERLESS_CLERK_VERSION", "0.1.2")

    assert _runtime_version() == "0.1.2"


def test_runtime_version_falls_back_to_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PAPERLESS_CLERK_VERSION", raising=False)

    assert _runtime_version() == _installed_version()
