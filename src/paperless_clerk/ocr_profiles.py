"""OCR request profiles.

A specialist OCR model is not an instruction-following chat model. Each one is
trained against a single short task command, and its serving stack may need a
few request fields of its own. A profile holds that complete contract in one
place, so the model client and the job record stay free of per-model branching.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

VLLM_FLAG = "CLERK_ENABLE_VLLM_PROFILES"
_TRUTHY = {"1", "true", "t", "yes", "y", "on"}

GENERIC_SYSTEM_PROMPT = (
    "You are an OCR engine. You transcribe document images to text verbatim. You never "
    "describe, summarize, or comment on a document, and you never withhold its contents."
)

# Written for an instruction-following vision model rather than a specialist OCR
# checkpoint. Each line answers a way those models spoil a document pipeline:
# they announce the transcription, they narrate the page instead of reading it,
# and they redact the account numbers and addresses that make a scan worth
# filing. The blank-page marker matters too, because an empty reply reads as a
# failed page and takes the whole document down with it.
GENERIC_PAGE_PROMPT = """Transcribe this page verbatim.

- Output the transcription only: no preamble, no closing remark, no code fence.
- Copy every character exactly, including names, addresses, account and policy
  numbers, dates, and amounts. Never redact, mask, or abbreviate them.
- Follow the page's reading order, taking multi-column layouts one full column
  at a time.
- Keep the line breaks of addresses, headings, labelled fields, and lists.
- Write tables as Markdown tables.
- Include headers, footers, page numbers, stamps, signatures, and handwriting.
  Write a ticked box as [x] and an empty one as [ ].
- Never translate, reword, correct, or complete anything on the page.
- Write [illegible] for text you cannot make out, and [blank page] on its own if
  the page carries no text at all."""


@dataclass(frozen=True)
class OCRProfile:
    key: str
    label: str
    prompt: str
    system: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Held back behind VLLM_FLAG. See vllm_profiles_enabled().
    vllm: bool = False


PROFILES: tuple[OCRProfile, ...] = (
    OCRProfile(
        key="generic",
        label="Generic vision model",
        prompt=GENERIC_PAGE_PROMPT,
        system=GENERIC_SYSTEM_PROMPT,
    ),
    OCRProfile(
        key="deepseek_ocr",
        label="DeepSeek-OCR-2 via vLLM",
        prompt="Free OCR.",
        vllm=True,
        # vLLM's DeepSeek-OCR recipe is a matched pair. The server registers
        # NGramPerReqLogitsProcessor and every request supplies its window;
        # without the request half the model loops on dense pages until it
        # exhausts the output limit. The whitelisted ids are the two image
        # tokens, which legitimately repeat. Suppressing the loop needs the
        # model's control tokens to stay visible, so they are stripped from the
        # transcription afterwards instead of at the server.
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],
            },
        },
    ),
    OCRProfile(
        key="deepseek_ocr_llamacpp",
        label="DeepSeek-OCR-2 GGUF via llama.cpp",
        prompt="Free OCR.",
        # llama.cpp has no equivalent logits processor. This is the known-good
        # request for the GGUF build and is deliberately left alone.
        extra_body={"top_k": 1},
    ),
    OCRProfile(
        key="glm_ocr",
        label="GLM-OCR via vLLM",
        prompt="Text Recognition:",
        vllm=True,
    ),
)

PROFILES_BY_KEY: dict[str, OCRProfile] = {profile.key: profile for profile in PROFILES}
PROFILE_KEYS: tuple[str, ...] = tuple(PROFILES_BY_KEY)


def vllm_profiles_enabled() -> bool:
    """Are the vLLM OCR profiles on offer?

    Both of them transcribe a page and then fall into a decoder loop that runs
    to the output limit. That is a serving bug in vLLM's OCR paths, not a
    request Clerk can reshape, so they stay out of the UI until it is fixed
    upstream. Setting CLERK_ENABLE_VLLM_PROFILES brings them back for retesting.
    """

    return os.environ.get(VLLM_FLAG, "").strip().casefold() in _TRUTHY


def available_profiles() -> tuple[OCRProfile, ...]:
    if vllm_profiles_enabled():
        return PROFILES
    return tuple(profile for profile in PROFILES if not profile.vllm)


def ocr_profile(key: str) -> OCRProfile:
    profile = PROFILES_BY_KEY.get(key)
    if profile is None:
        return PROFILES_BY_KEY["generic"]
    if profile.vllm and not vllm_profiles_enabled():
        # A profile chosen before it was held back must not stop the container
        # from starting, so fall back to the one contract that suits any server.
        log.warning(
            "OCR profile %r is disabled, so this run uses 'generic' instead. "
            "Set %s=1 to re-enable it.",
            key,
            VLLM_FLAG,
        )
        return PROFILES_BY_KEY["generic"]
    return profile
