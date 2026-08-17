from __future__ import annotations

import pytest

from paperless_clerk.ocr_profiles import VLLM_FLAG


@pytest.fixture
def vllm_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offer the vLLM OCR profiles, which are held back by default."""

    monkeypatch.setenv(VLLM_FLAG, "1")
