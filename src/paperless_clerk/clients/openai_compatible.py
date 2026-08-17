from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any

import httpx

from paperless_clerk.config import Settings
from paperless_clerk.ocr_profiles import ocr_profile
from paperless_clerk.rendering import render_ocr_test_image

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^\s*```[a-z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
# DeepSeek's grounded output labels each block with a layout class and a
# coordinate box. Both describe the page; neither is page text.
_GROUNDING_ANNOTATION = re.compile(r"<\|(ref|det)\|>.*?<\|/\1\|>", re.DOTALL)
# A profile that keeps special tokens visible leaves control tokens in the
# content. DeepSeek writes some of them with full-width pipes.
_SPECIAL_TOKEN = re.compile(r"<[|｜][^<>\n]{0,64}?[|｜]>")
_BLANK_RUN = re.compile(r"\n(?:[ \t]*\n){2,}")
_REFUSAL_PREFIXES = ("i cannot", "i can't", "i am unable", "i'm unable", "sorry, i")
# A cycle has to repeat this many times before we treat it as a decoder loop
# rather than a page that genuinely repeats a line.
_LOOP_MIN_REPEATS = 3

log = logging.getLogger(__name__)


def clean_ocr_text(text: str) -> str:
    """Remove response scaffolding without ever rewriting transcribed text."""

    text = _THINK_BLOCK.sub("", text)
    text = _CODE_FENCE.sub(r"\1", text)
    text = _GROUNDING_ANNOTATION.sub("", text)
    text = _SPECIAL_TOKEN.sub("", text)
    return _BLANK_RUN.sub("\n\n", text).strip()


def _drop_repeated_tail(units: list[str], max_period: int) -> list[str] | None:
    """Find a block repeated to the very end and return the text up to one copy."""

    for period in range(1, min(max_period, len(units) // _LOOP_MIN_REPEATS) + 1):
        block = units[-period:]
        index = len(units) - period
        repeats = 1
        while index >= period and units[index - period : index] == block:
            repeats += 1
            index -= period
        if repeats >= _LOOP_MIN_REPEATS:
            return units[: index + period]
    return None


def trim_runaway_repetition(text: str) -> str:
    """Cut a decoder loop off the end of a truncated page.

    A greedy decoder that falls into a cycle emits the same block until it hits
    the token cap, but the transcription before the cycle is still good. Keep one
    copy of the block and drop the rest. Only ever called on a truncated
    response, so a page that legitimately repeats a line is never touched.
    """

    stripped = text.strip()
    # A loop with no newline in it is one long line, so fall back to words.
    for units, joiner, max_period in (
        (stripped.splitlines(), "\n", 40),
        (stripped.split(" "), " ", 60),
    ):
        trimmed = _drop_repeated_tail(units, max_period)
        if trimmed is None:
            continue
        kept = joiner.join(trimmed).rstrip()
        # Only believe it is a decoder cycle when the repetition dominates the
        # response. A few identical table rows or repeated form labels are page
        # content, and deleting them would lose real text.
        if len(kept) <= 0.5 * len(stripped):
            return kept
    return text


class ModelError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class OpenAICompatibleClient:
    def __init__(self, settings: Settings, purpose: str):
        if purpose not in {"ocr", "metadata"}:
            raise ValueError("purpose must be ocr or metadata")
        self.purpose = purpose
        self.base_url = settings.openai_base_url.rstrip("/")
        self.model = getattr(settings, f"{purpose}_model")
        self.api_key = settings.secret_value("openai_api_key")
        self.max_output_tokens = getattr(settings, f"{purpose}_max_output_tokens")
        self.max_retries = settings.model_max_retries
        self.profile = ocr_profile(settings.ocr_profile if purpose == "ocr" else "generic")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds), headers=headers
        )

    async def close(self) -> None:
        await self.client.aclose()

    @property
    def completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    async def _post(
        self, payload: dict[str, Any], *, allow_format_fallback: bool = False
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        format_fallback_stage = 0
        attempt = 0
        while attempt <= self.max_retries:
            try:
                response = await self.client.post(self.completions_url, json=payload)
                if response.status_code < 400:
                    body = response.json()
                    if not isinstance(body, dict):
                        raise ValueError("response body must be a JSON object")
                    return body
                if (
                    allow_format_fallback
                    and response.status_code in {400, 404, 422}
                    and format_fallback_stage < 2
                ):
                    format_fallback_stage += 1
                    payload = dict(payload)
                    if format_fallback_stage == 1:
                        payload["response_format"] = {"type": "json_object"}
                    else:
                        payload.pop("response_format", None)
                    continue
                retryable = response.status_code in {408, 425, 429, 500, 502, 503, 504}
                message = response.text[:1200]
                if not retryable or attempt >= self.max_retries:
                    raise ModelError(
                        f"{self.purpose} model returned {response.status_code}: {message}",
                        retryable=retryable,
                    )
                retry_after = response.headers.get("retry-after")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(12, 0.75 * (2**attempt))
                )
                await asyncio.sleep(delay)
                attempt += 1
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise ModelError(
                        f"{self.purpose} model request failed: {exc}", retryable=True
                    ) from exc
                await asyncio.sleep(min(12, 0.75 * (2**attempt)))
                attempt += 1
            except (ValueError, json.JSONDecodeError) as exc:
                if attempt >= self.max_retries:
                    raise ModelError(
                        f"{self.purpose} model returned invalid response JSON: {exc}",
                        retryable=True,
                    ) from exc
                await asyncio.sleep(min(4, 0.5 * (2**attempt)))
                attempt += 1
        raise ModelError(f"{self.purpose} model request failed: {last_error}", retryable=True)

    @staticmethod
    def _content(body: dict[str, Any]) -> str:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(
                "Model response did not contain choices[0].message.content", retryable=True
            ) from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ModelError("Model returned empty content", retryable=True)
        return content.strip()

    @staticmethod
    def _finish_reason(body: dict[str, Any]) -> str | None:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        reason = choices[0].get("finish_reason")
        return str(reason) if reason is not None else None

    @staticmethod
    def _usage(body: dict[str, Any]) -> tuple[int | None, int | None]:
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return None, None
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        return (
            prompt if isinstance(prompt, int) else None,
            completion if isinstance(completion, int) else None,
        )

    def _truncation_detail(self, body: dict[str, Any]) -> str:
        """Explain a truncated page from the token counts rather than guessing.

        Stopping short of the requested cap means the server shortened it to fit
        the context, which is a completely different fix from a model that
        genuinely ran long, so never report one as the other.
        """

        prompt, completion = self._usage(body)
        if completion is None:
            return (
                f"The server reported no token usage, so the cap it applied is unknown "
                f"(this page requested {self.max_output_tokens})."
            )
        counts = f"prompt {prompt} + output {completion} tokens"
        if completion < 0.95 * self.max_output_tokens:
            room = f"{prompt + completion}" if prompt is not None else "the total"
            return (
                f"The server capped output at {completion} tokens even though "
                f"{self.max_output_tokens} were requested ({counts}), so the image filled the "
                f"context. Raise the server's context size above {room}, or lower the render DPI."
            )
        return (
            f"The model produced the full {completion} tokens requested ({counts}) for one page, "
            "which usually means it repeated itself rather than ran long."
        )

    async def ocr_page(self, image: bytes, *, page_number: int) -> str:
        encoded = base64.b64encode(image).decode("ascii")
        media_type = "image/png" if image.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
        messages: list[dict[str, Any]] = []
        if self.profile.system:
            messages.append({"role": "system", "content": self.profile.system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                    },
                    {"type": "text", "text": self.profile.prompt},
                ],
            }
        )
        body = await self._post(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": self.max_output_tokens,
                "messages": messages,
                **self.profile.extra_body,
            }
        )
        truncated = self._finish_reason(body) == "length"
        raw = self._content(body)
        text = clean_ocr_text(raw)
        if truncated:
            # A page that ran out of tokens still transcribed everything before
            # the cut, so keep that rather than failing the whole document. The
            # document-level meaningful-text check is what decides whether the
            # result is fit to publish.
            kept = trim_runaway_repetition(text)
            detail = self._truncation_detail(body)
            if not kept:
                raise ModelError(
                    f"OCR page {page_number} was truncated with no usable text. {detail}"
                )
            log.warning(
                "OCR page %s was truncated; keeping %s of %s characters. %s",
                page_number,
                len(kept),
                len(text),
                detail,
            )
            text = kept
        if not text:
            raise ModelError(
                f"OCR model returned no transcription for page {page_number}. "
                f"Raw response: {raw[:300]}"
            )
        if text.casefold().startswith(_REFUSAL_PREFIXES):
            raise ModelError(f"OCR model refused to transcribe page {page_number}: {text[:200]}")
        return text

    async def structured(
        self,
        *,
        name: str,
        schema: dict[str, Any],
        system: str,
        user: str,
    ) -> dict[str, Any]:
        schema_json = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        schema_instruction = (
            "Return exactly one JSON object matching the following JSON Schema. "
            "Use the property names exactly as written, include no additional properties, "
            "and do not replace nested objects with shorthand IDs or strings.\n"
            f"JSON Schema: {schema_json}"
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": f"{system}\n\n{schema_instruction}"},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        }
        body = await self._post(payload, allow_format_fallback=True)
        if self._finish_reason(body) in {"length", "max_tokens"}:
            raise ModelError("Structured metadata output reached the configured token limit")
        content = self._content(body)
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelError(
                f"Model did not return valid structured JSON: {exc}", retryable=True
            ) from exc
        if not isinstance(result, dict):
            raise ModelError("Structured model response must be a JSON object", retryable=True)
        return result

    async def test_connection(self) -> dict[str, Any]:
        if self.purpose == "ocr":
            # Send a real rendered page through the production request so the
            # test exercises the vision path and the selected profile, then
            # hand the transcription back for a human to judge.
            text = await self.ocr_page(render_ocr_test_image(), page_number=1)
            normalized = re.sub(r"\s+", " ", text.casefold())
            if not any(
                marker in normalized
                for marker in ("paperless clerk", "4827", "end of clerk ocr test")
            ):
                raise ModelError(
                    "OCR model responded but transcribed none of the test page's known text. "
                    f"Check the model and profile. Response: {text[:200]}"
                )
            return {
                "ok": True,
                "model": self.model,
                "profile": self.profile.key,
                "response": text[:300],
            }
        body = await self._post(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Reply with OK."}],
            }
        )
        return {"ok": True, "model": self.model, "response": self._content(body)[:80]}
