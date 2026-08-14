from __future__ import annotations

import asyncio
import base64
import json
import re
from typing import Any

import httpx

from paperless_clerk.config import Settings
from paperless_clerk.prompts import DEEPSEEK_OCR_PAGE_PROMPT
from paperless_clerk.rendering import render_ocr_test_image


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
        self.ocr_profile = settings.ocr_profile if purpose == "ocr" else "generic"
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

    async def ocr_page(self, image: bytes, *, page_number: int, prompt: str) -> str:
        encoded = base64.b64encode(image).decode("ascii")
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        }
        if self.ocr_profile == "deepseek_ocr":
            # DeepSeek-OCR and DeepSeek-OCR-2 share this plain-text OCR task.
            # Grounded Markdown mode emits layout/coordinate artifacts that do
            # not belong in Paperless's canonical OCR text field.
            messages = [
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {"type": "text", "text": DEEPSEEK_OCR_PAGE_PROMPT},
                    ],
                }
            ]
            temperature = 0
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are the OCR engine for Paperless Clerk. Transcribe faithfully; never infer missing text.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Page {page_number}. {prompt}"},
                        image_content,
                    ],
                },
            ]
            temperature = 0
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
            "messages": messages,
        }
        if self.ocr_profile != "generic":
            # llama.cpp recommends greedy-ish decoding for DeepSeek OCR.
            payload["top_k"] = 1
        body = await self._post(payload)
        if self._finish_reason(body) in {"length", "max_tokens"}:
            raise ModelError(
                f"OCR output for page {page_number} reached the configured token limit"
            )
        text = self._content(body)
        text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(
            r"^\s*```(?:text|markdown)?\s*\n?(.*?)\n?```\s*$",
            r"\1",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if self.ocr_profile == "deepseek_ocr":
            # Be defensive if a DeepSeek server returns grounded annotations
            # despite the plain OCR prompt: retain recognized text, discard only
            # the model-specific reference and coordinate scaffolding.
            text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
            text = re.sub(r"<\|/?ref\|>", "", text)
            text = text.replace("<|grounding|>", "")
        if not text.strip():
            raise ModelError(
                "OCR model returned no transcription after removing response scaffolding"
            )
        refusal = text.strip().casefold()
        if refusal.startswith(("i cannot", "i can't", "i am unable", "i'm unable", "sorry, i")):
            raise ModelError(f"OCR model refused to transcribe page {page_number}")
        return text.strip()

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
            text = await self.ocr_page(
                render_ocr_test_image(),
                page_number=1,
                prompt="Transcribe the clearly printed test text.",
            )
            if "clerk" not in text.casefold():
                raise ModelError(
                    "OCR model responded but did not read the test image; check the selected "
                    "profile and matching multimodal projector"
                )
            return {
                "ok": True,
                "model": self.model,
                "profile": self.ocr_profile,
                "response": text[:80],
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
